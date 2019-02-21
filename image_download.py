# https://github.com/hardikvasa/google-images-download

from google_images_download import google_images_download
from os import listdir

source_image_dir = "../food-101/images/"
folders = [i for i in listdir(source_image_dir)]

response = google_images_download.googleimagesdownload()

for food in folders:
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
    print(paths)