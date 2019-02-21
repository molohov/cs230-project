# docmentation for google_images_download API can be found at
# https://github.com/hardikvasa/google-images-download

from google_images_download import google_images_download
from os import listdir

# grab the source image dir for the labels to download additional data for
source_image_dir = "../food-101/images/"

response = google_images_download.googleimagesdownload()

for food in listdir(source_image_dir):
    food_keyword = food
    food_keyword.replace('_', ' ')
    arguments = {
        "keywords" : food_keyword,
        "limit" : 100,
        "print_urls" : True,
        "no_numbering" : True,
        "safe_search" : True,
        "image_directory" : food,
        "output_directory" : "image_download",
    }

    paths = response.download(arguments)