import os
import torch
import pandas as pd
import numpy as np
from torchvision.transforms import ToPILImage
from transformers import AutoImageProcessor

from flmr import index_custom_collection
from flmr import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval
from transformers import BertTokenizer
# load models
checkpoint_path = "LinWeizheDragon/PreFLMR_ViT-G"
image_processor_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

# query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="query_tokenizer")
# context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
#     checkpoint_path, subfolder="context_tokenizer"
# )
base_tknzer = BertTokenizer.from_pretrained("bert-base-uncased")

model = FLMRModelForRetrieval.from_pretrained(
    checkpoint_path,
    query_tokenizer=base_tknzer,
    context_tokenizer=base_tknzer,
)
image_processor = AutoImageProcessor.from_pretrained(image_processor_name)


num_items = 100
feature_dim = 1664
passage_contents = [f"This is test sentence {n}" for n in range(num_items) ]
# Option 1. text-only documents
custom_collection = passage_contents
# Option 2. multi-modal documents with pre-extracted image features
# passage_image_features = np.random.rand(num_items, feature_dim)
# custom_collection = [
#     (passage_content, passage_image_feature, None) for passage_content, passage_image_feature in zip(passage_contents, passage_image_features)
# ]
# Option 3. multi-modal documents with images
# random_images = torch.randn(num_items, 3, 224, 224)
# to_img = ToPILImage()
# if not os.path.exists("./test_images"):
#     os.makedirs("./test_images")
# for i, image in enumerate(random_images):
#     image = to_img(image)
#     image.save(os.path.join("./test_images", "{}.jpg".format(i)))

# image_paths = [os.path.join("./test_images", "{}.jpg".format(i)) for i in range(num_items)]

# custom_collection = [
#     (passage_content, None, image_path)
#     for passage_content, image_path in zip(passage_contents, image_paths)
# ]


if __name__ == "__main__":
    # freeze_support()
    index_custom_collection(
        custom_collection=custom_collection,
        model=checkpoint_path,
        index_root_path=".",
        index_experiment_name="test_experiment",
        index_name="test_index",
        nbits=8, # number of bits in compression
        doc_maxlen=512, # maximum allowed document length
        overwrite=True, # whether to overwrite existing indices
        use_gpu=True, # whether to enable GPU indexing
        indexing_batch_size=64,
        model_temp_folder="./tmp/",
        nranks=1, # number of GPUs used in indexing
    )