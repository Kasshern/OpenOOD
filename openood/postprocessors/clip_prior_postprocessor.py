"""
CLIPPriorPostprocessor: uses CLIP as a Bayesian prior for OOD detection.

Bayesian framing:
  - CLIP provides log P(class | image) via zero-shot image-text similarity
    (broad prior over N classes, trained on internet-scale data)
  - The base model net(data) provides log P(class | image) from supervised
    training on the ID dataset (specialized likelihood)
  - Combined posterior in log-space:
      combined_logits = alpha * (clip_logits * logit_scale) + (1 - alpha) * base_logits
  - OOD score (higher = more in-distribution):
      softmax mode:  conf = softmax(combined_logits).max()
      energy mode:   conf = logsumexp(combined_logits)

Dataset leakage note:
  CLIP was pre-trained on ~400M web image-text pairs. ImageNet images
  almost certainly appear in that data. Results on ImageNet benchmarks
  should be interpreted cautiously. Far-OOD benchmarks (iNaturalist, SUN,
  Places, Textures) are more informative. CIFAR leakage risk is lower.

Supported ID datasets: cifar10, cifar100, imagenet200, imagenet.
"""

import os
import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from openood.preprocessors.transform import normalization_dict
from openood.networks.clip import CLIPZeroshot

from .base_postprocessor import BasePostprocessor

# ── CLIP normalization (target, dataset-independent) ──────────────────────────
_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# ── Class name lists ──────────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

CIFAR100_CLASSES = [
    'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
    'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
    'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
    'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
    'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
    'maple tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak tree',
    'orange', 'orchid', 'otter', 'palm tree', 'pear', 'pickup truck',
    'pine tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit',
    'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
    'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
    'streetcar', 'sunflower', 'sweet pepper', 'table', 'tank', 'telephone',
    'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
    'wardrobe', 'whale', 'willow tree', 'wolf', 'woman', 'worm',
]

# 1000 ImageNet class names in ILSVRC index order.
# Copied from scripts/eval_ood_imagenet_foundation_models.py.
IMAGENET_CLASSES = [
    'tench', 'goldfish', 'great white shark', 'tiger shark',
    'hammerhead shark', 'electric ray', 'stingray', 'rooster', 'hen',
    'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco',
    'indigo bunting', 'American robin', 'bulbul', 'jay', 'magpie', 'chickadee',
    'American dipper', 'kite (bird of prey)', 'bald eagle', 'vulture',
    'great grey owl', 'fire salamander', 'smooth newt', 'newt',
    'spotted salamander', 'axolotl', 'American bullfrog', 'tree frog',
    'tailed frog', 'loggerhead sea turtle', 'leatherback sea turtle',
    'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'green iguana',
    'Carolina anole', 'desert grassland whiptail lizard', 'agama',
    'frilled-necked lizard', 'alligator lizard', 'Gila monster',
    'European green lizard', 'chameleon', 'Komodo dragon', 'Nile crocodile',
    'American alligator', 'triceratops', 'worm snake', 'ring-necked snake',
    'eastern hog-nosed snake', 'smooth green snake', 'kingsnake',
    'garter snake', 'water snake', 'vine snake', 'night snake',
    'boa constrictor', 'African rock python', 'Indian cobra', 'green mamba',
    'sea snake', 'Saharan horned viper', 'eastern diamondback rattlesnake',
    'sidewinder rattlesnake', 'trilobite', 'harvestman', 'scorpion',
    'yellow garden spider', 'barn spider', 'European garden spider',
    'southern black widow', 'tarantula', 'wolf spider', 'tick', 'centipede',
    'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie grouse', 'peafowl',
    'quail', 'partridge', 'african grey parrot', 'macaw',
    'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill',
    'hummingbird', 'jacamar', 'toucan', 'duck', 'red-breasted merganser',
    'goose', 'black swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala',
    'wombat', 'jellyfish', 'sea anemone', 'brain coral', 'flatworm',
    'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton',
    'chambered nautilus', 'Dungeness crab', 'rock crab', 'fiddler crab',
    'red king crab', 'American lobster', 'spiny lobster', 'crayfish',
    'hermit crab', 'isopod', 'white stork', 'black stork', 'spoonbill',
    'flamingo', 'little blue heron', 'great egret', 'bittern bird',
    'crane bird', 'limpkin', 'common gallinule', 'American coot', 'bustard',
    'ruddy turnstone', 'dunlin', 'common redshank', 'dowitcher',
    'oystercatcher', 'pelican', 'king penguin', 'albatross', 'grey whale',
    'killer whale', 'dugong', 'sea lion', 'Chihuahua', 'Japanese Chin',
    'Maltese', 'Pekingese', 'Shih Tzu', 'King Charles Spaniel', 'Papillon',
    'toy terrier', 'Rhodesian Ridgeback', 'Afghan Hound', 'Basset Hound',
    'Beagle', 'Bloodhound', 'Bluetick Coonhound', 'Black and Tan Coonhound',
    'Treeing Walker Coonhound', 'English foxhound', 'Redbone Coonhound',
    'borzoi', 'Irish Wolfhound', 'Italian Greyhound', 'Whippet',
    'Ibizan Hound', 'Norwegian Elkhound', 'Otterhound', 'Saluki',
    'Scottish Deerhound', 'Weimaraner', 'Staffordshire Bull Terrier',
    'American Staffordshire Terrier', 'Bedlington Terrier', 'Border Terrier',
    'Kerry Blue Terrier', 'Irish Terrier', 'Norfolk Terrier',
    'Norwich Terrier', 'Yorkshire Terrier', 'Wire Fox Terrier',
    'Lakeland Terrier', 'Sealyham Terrier', 'Airedale Terrier',
    'Cairn Terrier', 'Australian Terrier', 'Dandie Dinmont Terrier',
    'Boston Terrier', 'Miniature Schnauzer', 'Giant Schnauzer',
    'Standard Schnauzer', 'Scottish Terrier', 'Tibetan Terrier',
    'Australian Silky Terrier', 'Soft-coated Wheaten Terrier',
    'West Highland White Terrier', 'Lhasa Apso', 'Flat-Coated Retriever',
    'Curly-coated Retriever', 'Golden Retriever', 'Labrador Retriever',
    'Chesapeake Bay Retriever', 'German Shorthaired Pointer', 'Vizsla',
    'English Setter', 'Irish Setter', 'Gordon Setter', 'Brittany dog',
    'Clumber Spaniel', 'English Springer Spaniel', 'Welsh Springer Spaniel',
    'Cocker Spaniel', 'Sussex Spaniel', 'Irish Water Spaniel', 'Kuvasz',
    'Schipperke', 'Groenendael dog', 'Malinois', 'Briard', 'Australian Kelpie',
    'Komondor', 'Old English Sheepdog', 'Shetland Sheepdog', 'collie',
    'Border Collie', 'Bouvier des Flandres dog', 'Rottweiler',
    'German Shepherd Dog', 'Dobermann', 'Miniature Pinscher',
    'Greater Swiss Mountain Dog', 'Bernese Mountain Dog',
    'Appenzeller Sennenhund', 'Entlebucher Sennenhund', 'Boxer', 'Bullmastiff',
    'Tibetan Mastiff', 'French Bulldog', 'Great Dane', 'St. Bernard', 'husky',
    'Alaskan Malamute', 'Siberian Husky', 'Dalmatian', 'Affenpinscher',
    'Basenji', 'pug', 'Leonberger', 'Newfoundland dog', 'Great Pyrenees dog',
    'Samoyed', 'Pomeranian', 'Chow Chow', 'Keeshond', 'brussels griffon',
    'Pembroke Welsh Corgi', 'Cardigan Welsh Corgi', 'Toy Poodle',
    'Miniature Poodle', 'Standard Poodle',
    'Mexican hairless dog (xoloitzcuintli)', 'grey wolf',
    'Alaskan tundra wolf', 'red wolf or maned wolf', 'coyote', 'dingo',
    'dhole', 'African wild dog', 'hyena', 'red fox', 'kit fox', 'Arctic fox',
    'grey fox', 'tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat',
    'Egyptian Mau', 'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar',
    'lion', 'tiger', 'cheetah', 'brown bear', 'American black bear',
    'polar bear', 'sloth bear', 'mongoose', 'meerkat', 'tiger beetle',
    'ladybug', 'ground beetle', 'longhorn beetle', 'leaf beetle',
    'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant',
    'grasshopper', 'cricket insect', 'stick insect', 'cockroach',
    'praying mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly',
    'damselfly', 'red admiral butterfly', 'ringlet butterfly',
    'monarch butterfly', 'small white butterfly', 'sulphur butterfly',
    'gossamer-winged butterfly', 'starfish', 'sea urchin', 'sea cucumber',
    'cottontail rabbit', 'hare', 'Angora rabbit', 'hamster', 'porcupine',
    'fox squirrel', 'marmot', 'beaver', 'guinea pig', 'common sorrel horse',
    'zebra', 'pig', 'wild boar', 'warthog', 'hippopotamus', 'ox',
    'water buffalo', 'bison', 'ram (adult male sheep)', 'bighorn sheep',
    'Alpine ibex', 'hartebeest', 'impala (antelope)', 'gazelle',
    'arabian camel', 'llama', 'weasel', 'mink', 'European polecat',
    'black-footed ferret', 'otter', 'skunk', 'badger', 'armadillo',
    'three-toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon',
    'siamang', 'guenon', 'patas monkey', 'baboon', 'macaque', 'langur',
    'black-and-white colobus', 'proboscis monkey', 'marmoset',
    'white-headed capuchin', 'howler monkey', 'titi monkey',
    "Geoffroy's spider monkey", 'common squirrel monkey', 'ring-tailed lemur',
    'indri', 'Asian elephant', 'African bush elephant', 'red panda',
    'giant panda', 'snoek fish', 'eel', 'silver salmon', 'rock beauty fish',
    'clownfish', 'sturgeon', 'gar fish', 'lionfish', 'pufferfish', 'abacus',
    'abaya', 'academic gown', 'accordion', 'acoustic guitar',
    'aircraft carrier', 'airliner', 'airship', 'altar', 'ambulance',
    'amphibious vehicle', 'analog clock', 'apiary', 'apron', 'trash can',
    'assault rifle', 'backpack', 'bakery', 'balance beam', 'balloon',
    'ballpoint pen', 'Band-Aid', 'banjo', 'baluster / handrail', 'barbell',
    'barber chair', 'barbershop', 'barn', 'barometer', 'barrel', 'wheelbarrow',
    'baseball', 'basketball', 'bassinet', 'bassoon', 'swimming cap',
    'bath towel', 'bathtub', 'station wagon', 'lighthouse', 'beaker',
    'military hat (bearskin or shako)', 'beer bottle', 'beer glass',
    'bell tower', 'baby bib', 'tandem bicycle', 'bikini', 'ring binder',
    'binoculars', 'birdhouse', 'boathouse', 'bobsleigh', 'bolo tie',
    'poke bonnet', 'bookcase', 'bookstore', 'bottle cap', 'hunting bow',
    'bow tie', 'brass memorial plaque', 'bra', 'breakwater', 'breastplate',
    'broom', 'bucket', 'buckle', 'bulletproof vest', 'high-speed train',
    'butcher shop', 'taxicab', 'cauldron', 'candle', 'cannon', 'canoe',
    'can opener', 'cardigan', 'car mirror', 'carousel', 'tool kit',
    'cardboard box / carton', 'car wheel', 'automated teller machine',
    'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello',
    'mobile phone', 'chain', 'chain-link fence', 'chain mail', 'chainsaw',
    'storage chest', 'chiffonier', 'bell or wind chime', 'china cabinet',
    'Christmas stocking', 'church', 'movie theater', 'cleaver',
    'cliff dwelling', 'cloak', 'clogs', 'cocktail shaker', 'coffee mug',
    'coffeemaker', 'spiral or coil', 'combination lock', 'computer keyboard',
    'candy store', 'container ship', 'convertible', 'corkscrew', 'cornet',
    'cowboy boot', 'cowboy hat', 'cradle', 'construction crane',
    'crash helmet', 'crate', 'infant bed', 'Crock Pot', 'croquet ball',
    'crutch', 'cuirass', 'dam', 'desk', 'desktop computer',
    'rotary dial telephone', 'diaper', 'digital clock', 'digital watch',
    'dining table', 'dishcloth', 'dishwasher', 'disc brake', 'dock',
    'dog sled', 'dome', 'doormat', 'drilling rig', 'drum', 'drumstick',
    'dumbbell', 'Dutch oven', 'electric fan', 'electric guitar',
    'electric locomotive', 'entertainment center', 'envelope',
    'espresso machine', 'face powder', 'feather boa', 'filing cabinet',
    'fireboat', 'fire truck', 'fire screen', 'flagpole', 'flute',
    'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen',
    'four-poster bed', 'freight car', 'French horn', 'frying pan', 'fur coat',
    'garbage truck', 'gas mask or respirator', 'gas pump', 'goblet', 'go-kart',
    'golf ball', 'golf cart', 'gondola', 'gong', 'gown', 'grand piano',
    'greenhouse', 'radiator grille', 'grocery store', 'guillotine',
    'hair clip', 'hair spray', 'half-track', 'hammer', 'hamper', 'hair dryer',
    'hand-held computer', 'handkerchief', 'hard disk drive', 'harmonica',
    'harp', 'combine harvester', 'hatchet', 'holster', 'home theater',
    'honeycomb', 'hook', 'hoop skirt', 'gymnastic horizontal bar',
    'horse-drawn vehicle', 'hourglass', 'iPod', 'clothes iron',
    'carved pumpkin', 'jeans', 'jeep', 'T-shirt', 'jigsaw puzzle', 'rickshaw',
    'joystick', 'kimono', 'knee pad', 'knot', 'lab coat', 'ladle', 'lampshade',
    'laptop computer', 'lawn mower', 'lens cap', 'letter opener', 'library',
    'lifeboat', 'lighter', 'limousine', 'ocean liner', 'lipstick',
    'slip-on shoe', 'lotion', 'music speaker', 'loupe magnifying glass',
    'sawmill', 'magnetic compass', 'messenger bag', 'mailbox', 'tights',
    'one-piece bathing suit', 'manhole cover', 'maraca', 'marimba', 'mask',
    'matchstick', 'maypole', 'maze', 'measuring cup', 'medicine cabinet',
    'megalith', 'microphone', 'microwave oven', 'military uniform', 'milk can',
    'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl',
    'mobile home', 'ford model t', 'modem', 'monastery', 'monitor', 'moped',
    'mortar and pestle', 'graduation cap', 'mosque', 'mosquito net', 'vespa',
    'mountain bike', 'tent', 'computer mouse', 'mousetrap', 'moving van',
    'muzzle', 'metal nail', 'neck brace', 'necklace', 'baby pacifier',
    'notebook computer', 'obelisk', 'oboe', 'ocarina', 'odometer',
    'oil filter', 'pipe organ', 'oscilloscope', 'overskirt', 'bullock cart',
    'oxygen mask', 'product packet / packaging', 'paddle', 'paddle wheel',
    'padlock', 'paintbrush', 'pajamas', 'palace', 'pan flute', 'paper towel',
    'parachute', 'parallel bars', 'park bench', 'parking meter',
    'railroad car', 'patio', 'payphone', 'pedestal', 'pencil case',
    'pencil sharpener', 'perfume', 'Petri dish', 'photocopier', 'plectrum',
    'Pickelhaube', 'picket fence', 'pickup truck', 'pier', 'piggy bank',
    'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate ship',
    'drink pitcher', 'block plane', 'planetarium', 'plastic bag', 'plate rack',
    'farm plow', 'plunger', 'Polaroid camera', 'pole', 'police van', 'poncho',
    'pool table', 'soda bottle', 'plant pot', "potter's wheel", 'power drill',
    'prayer rug', 'printer', 'prison', 'missile', 'projector', 'hockey puck',
    'punching bag', 'purse', 'quill', 'quilt', 'race car', 'racket',
    'radiator', 'radio', 'radio telescope', 'rain barrel',
    'recreational vehicle', 'fishing casting reel', 'reflex camera',
    'refrigerator', 'remote control', 'restaurant', 'revolver', 'rifle',
    'rocking chair', 'rotisserie', 'eraser', 'rugby ball',
    'ruler measuring stick', 'sneaker', 'safe', 'safety pin', 'salt shaker',
    'sandal', 'sarong', 'saxophone', 'scabbard', 'weighing scale',
    'school bus', 'schooner', 'scoreboard', 'CRT monitor', 'screw',
    'screwdriver', 'seat belt', 'sewing machine', 'shield', 'shoe store',
    'shoji screen / room divider', 'shopping basket', 'shopping cart',
    'shovel', 'shower cap', 'shower curtain', 'ski', 'balaclava ski mask',
    'sleeping bag', 'slide rule', 'sliding door', 'slot machine', 'snorkel',
    'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball', 'sock',
    'solar thermal collector', 'sombrero', 'soup bowl', 'keyboard space bar',
    'space heater', 'space shuttle', 'spatula', 'motorboat', 'spider web',
    'spindle', 'sports car', 'spotlight', 'stage', 'steam locomotive',
    'through arch bridge', 'steel drum', 'stethoscope', 'scarf', 'stone wall',
    'stopwatch', 'stove', 'strainer', 'tram', 'stretcher', 'couch', 'stupa',
    'submarine', 'suit', 'sundial', 'sunglasses', 'sunglasses', 'sunscreen',
    'suspension bridge', 'mop', 'sweatshirt', 'swim trunks / shorts', 'swing',
    'electrical switch', 'syringe', 'table lamp', 'tank', 'tape player',
    'teapot', 'teddy bear', 'television', 'tennis ball', 'thatched roof',
    'front curtain', 'thimble', 'threshing machine', 'throne', 'tile roof',
    'toaster', 'tobacco shop', 'toilet seat', 'torch', 'totem pole',
    'tow truck', 'toy store', 'tractor', 'semi-trailer truck', 'tray',
    'trench coat', 'tricycle', 'trimaran', 'tripod', 'triumphal arch',
    'trolleybus', 'trombone', 'hot tub', 'turnstile', 'typewriter keyboard',
    'umbrella', 'unicycle', 'upright piano', 'vacuum cleaner', 'vase',
    'vaulted or arched ceiling', 'velvet fabric', 'vending machine',
    'vestment', 'viaduct', 'violin', 'volleyball', 'waffle iron', 'wall clock',
    'wallet', 'wardrobe', 'military aircraft', 'sink', 'washing machine',
    'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle',
    'hair wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle',
    'airplane wing', 'wok', 'wooden spoon', 'wool', 'split-rail fence',
    'shipwreck', 'sailboat', 'yurt', 'website', 'comic book', 'crossword',
    'traffic or street sign', 'traffic light', 'dust jacket', 'menu', 'plate',
    'guacamole', 'consomme', 'hot pot', 'trifle', 'ice cream', 'popsicle',
    'baguette', 'bagel', 'pretzel', 'cheeseburger', 'hot dog',
    'mashed potatoes', 'cabbage', 'broccoli', 'cauliflower', 'zucchini',
    'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber',
    'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith apple',
    'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit',
    'cherimoya (custard apple)', 'pomegranate', 'hay', 'carbonara',
    'chocolate syrup', 'dough', 'meatloaf', 'pizza', 'pot pie', 'burrito',
    'red wine', 'espresso', 'tea cup', 'eggnog', 'mountain', 'bubble', 'cliff',
    'coral reef', 'geyser', 'lakeshore', 'promontory', 'sandbar', 'beach',
    'valley', 'volcano', 'baseball player', 'bridegroom', 'scuba diver',
    'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'rose hip',
    'horse chestnut seed', 'coral fungus', 'agaric', 'gyromitra',
    'stinkhorn mushroom', 'earth star fungus', 'hen of the woods mushroom',
    'bolete', 'corn cob', 'toilet paper',
]

# 80 ImageNet prompt templates.
# Copied from scripts/eval_ood_imagenet_foundation_models.py.
IMAGENET_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

# ── ImageNet-200 class name resolution ───────────────────────────────────────

def _resolve_imagenet200_classnames(imglist_path: str, data_root: str) -> list:
    """
    Derive the 200 class names used by the OpenOOD imagenet200 benchmark.

    Reads the training imglist to extract (synset_id, label_int) pairs, then
    maps synset IDs → English names via torchvision's ImageNet meta file.

    Falls back to raw synset IDs as class names if the meta file is
    unavailable (CLIP will produce weaker text embeddings but won't crash).

    Args:
        imglist_path: Path to train_imagenet200.txt.
        data_root:    Root of the data directory (used to find imagenet/meta.bin).

    Returns:
        List of 200 class name strings, ordered by benchmark label index 0–199.

    Raises:
        FileNotFoundError: If imglist_path does not exist.
        ValueError:        If fewer than 200 unique labels are found.
    """
    if not os.path.exists(imglist_path):
        raise FileNotFoundError(
            f'[CLIPPrior] ImageNet-200 imglist not found: {imglist_path}\n'
            f'  Make sure data_root ({data_root}) points to the correct data '
            f'directory containing benchmark_imglist/imagenet200/.'
        )

    synset_re = re.compile(r'(n\d{8})')
    label_to_synset = {}
    with open(imglist_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            path, label = parts[0], int(parts[1])
            m = synset_re.search(path)
            if m and label not in label_to_synset:
                label_to_synset[label] = m.group(1)

    if len(label_to_synset) < 200:
        raise ValueError(
            f'[CLIPPrior] Only found {len(label_to_synset)} unique labels in '
            f'{imglist_path}; expected 200.'
        )

    # Build synset → English name via torchvision meta file
    synset_to_name = _load_synset_to_name(data_root)

    classnames = [''] * 200
    for label, synset in sorted(label_to_synset.items()):
        classnames[label] = synset_to_name.get(synset, synset)

    unnamed = [i for i, n in enumerate(classnames) if not n]
    if unnamed:
        print(f'[CLIPPrior] Warning: {len(unnamed)} imagenet200 classes have '
              f'no English name (using synset ID as fallback). '
              f'CLIP text embeddings for these will be suboptimal.')

    return classnames


def _load_synset_to_name(data_root: str) -> dict:
    """
    Return a dict mapping ILSVRC synset ID → English class name.

    Tries torchvision's load_meta_file (requires imagenet/meta.bin).
    Falls back to an empty dict so callers can use synset IDs as names.
    """
    meta_root = os.path.join(data_root, 'imagenet')
    try:
        import torchvision.datasets.imagenet as tv_imagenet
        wnids, _ = tv_imagenet.load_meta_file(meta_root)
        if len(wnids) == len(IMAGENET_CLASSES):
            return {wnid: IMAGENET_CLASSES[i] for i, wnid in enumerate(wnids)}
        else:
            print(f'[CLIPPrior] Warning: torchvision meta returned '
                  f'{len(wnids)} synsets (expected 1000); skipping name lookup.')
    except Exception as e:
        print(f'[CLIPPrior] Could not load imagenet synset meta from '
              f'{meta_root}: {e}\n'
              f'  Falling back to synset IDs as class names.')
    return {}


# ── Main postprocessor ────────────────────────────────────────────────────────

class CLIPPriorPostprocessor(BasePostprocessor):
    """
    OOD detector that combines a CLIP zero-shot prior with a base model.

    Combined logits:
        combined = alpha * (clip_logits * logit_scale) + (1 - alpha) * base_logits

    alpha=0 → pure base model (equivalent to MSP/energy on base model).
    alpha=1 → pure CLIP zero-shot.
    Optimal alpha is found via APS hyperparameter sweep on the val OOD set.
    """

    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep

        self.alpha      = float(self.args.alpha)
        self.score_mode = str(getattr(self.args, 'score_mode', 'softmax'))
        self.backbone   = str(getattr(self.args, 'clip_backbone', 'ViT-B/16'))
        self.data_root  = str(getattr(self.args, 'data_root', './data'))

        self.dataset_name = self.config.dataset.name

        self.clip_model    = None
        self._renorm_scale = None  # [1, 3, 1, 1] float32 on CUDA
        self._renorm_shift = None
        self.setup_flag    = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        classnames = self._resolve_classnames()
        print(f'[CLIPPrior] Loading CLIP {self.backbone} '
              f'with {len(classnames)} classes ({self.dataset_name})...')

        self.clip_model = CLIPZeroshot(
            classnames=classnames,
            templates=IMAGENET_TEMPLATES,
            backbone=self.backbone,
        )
        self.clip_model.cuda().eval()

        # Pre-compute per-dataset renorm constants (source stats → CLIP stats)
        if self.dataset_name not in normalization_dict:
            raise ValueError(
                f'[CLIPPrior] No normalization stats for dataset '
                f'"{self.dataset_name}". '
                f'Add it to openood/preprocessors/transform.py::normalization_dict.'
            )
        src_mean_v, src_std_v = normalization_dict[self.dataset_name]
        src_mean = torch.tensor(src_mean_v, dtype=torch.float32).view(1, 3, 1, 1).cuda()
        src_std  = torch.tensor(src_std_v,  dtype=torch.float32).view(1, 3, 1, 1).cuda()
        cl_mean  = torch.tensor(_CLIP_MEAN,  dtype=torch.float32).view(1, 3, 1, 1).cuda()
        cl_std   = torch.tensor(_CLIP_STD,   dtype=torch.float32).view(1, 3, 1, 1).cuda()

        self._renorm_scale = src_std / cl_std   # [1, 3, 1, 1]
        self._renorm_shift = (src_mean - cl_mean) / cl_std

        self.setup_flag = True

    def _resolve_classnames(self) -> list:
        name = self.dataset_name
        if name == 'cifar10':
            return CIFAR10_CLASSES
        elif name == 'cifar100':
            return CIFAR100_CLASSES
        elif name == 'imagenet':
            return IMAGENET_CLASSES
        elif name == 'imagenet200':
            imglist_path = os.path.join(
                self.data_root,
                'benchmark_imglist', 'imagenet200', 'train_imagenet200.txt',
            )
            return _resolve_imagenet200_classnames(imglist_path, self.data_root)
        else:
            raise ValueError(
                f'[CLIPPrior] Unsupported dataset "{name}". '
                f'Supported: cifar10, cifar100, imagenet200, imagenet.'
            )

    # ── Postprocess ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # 1. Base model forward
        base_logits = net(data)                          # [B, C]

        # 2. Re-normalize for CLIP (affine channel-wise transform)
        clip_data = data.float() * self._renorm_scale + self._renorm_shift

        # 3. Upsample if needed (CIFAR 32×32 → CLIP 224×224)
        if clip_data.shape[-1] != 224:
            clip_data = F.interpolate(
                clip_data, size=224, mode='bicubic', align_corners=False,
            )

        # 4. CLIP forward
        clip_logits = self.clip_model(clip_data)         # [B, C], cosine sims
        clip_logits = clip_logits.to(base_logits.dtype)

        # 5. Scale CLIP logits to match base model magnitude
        logit_scale = self.clip_model.model.logit_scale.exp().item()

        # 6. Bayesian combination in log-space
        combined = (
            self.alpha * (clip_logits * logit_scale)
            + (1.0 - self.alpha) * base_logits
        )                                                 # [B, C]

        # 7. OOD score
        if self.score_mode == 'softmax':
            prob = torch.softmax(combined, dim=1)
            conf, pred = torch.max(prob, dim=1)
        elif self.score_mode == 'energy':
            conf = torch.logsumexp(combined, dim=1)
            pred = combined.argmax(dim=1)
        else:
            raise ValueError(
                f'[CLIPPrior] Unknown score_mode "{self.score_mode}". '
                f'Use "softmax" or "energy".'
            )

        return pred, conf

    # ── APS hyperparameter interface ─────────────────────────────────────────

    def set_hyperparam(self, hyperparam: list):
        """Called by Evaluator.hyperparam_search(); hyperparam = [alpha]."""
        self.alpha = hyperparam[0]

    def get_hyperparam(self):
        return [self.alpha]
