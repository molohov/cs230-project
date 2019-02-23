import json
import master_config

with open(master_config.train_dict_loc,'r') as inf:
    dict_from_file = eval(inf.read())

partition = {}
labels = {}
class_id_indexes = {}
class_id_counter = 0
for class_id in dict_from_file:
    image_ids = dict_from_file[class_id]
    new_image_ids = []
    for img in image_ids:
        new_image_ids.append(class_id+"/"+img)
    if 'train' in partition:
        partition['train'].extend(new_image_ids)
    else:
        partition['train'] = new_image_ids
    for image in new_image_ids:
        if class_id in class_id_indexes:
            labels[image] = class_id_indexes[class_id]
        else:
            class_id_indexes[class_id] = class_id_counter
            labels[image] = class_id_counter
            class_id_counter = class_id_counter + 1

with open(master_config.dev_dict_loc,'r') as inf:
    dict_from_file = eval(inf.read())

for class_id in dict_from_file:
    image_ids = dict_from_file[class_id]
    new_image_ids = []
    for img in image_ids:
        new_image_ids.append(class_id + "/" + img)
    if 'validation' in partition:
        partition['validation'].extend(new_image_ids)
    else:
        partition['validation'] = new_image_ids
    for image in new_image_ids:
        labels[image] = class_id_indexes[class_id]

with open('partition.dict', 'w') as file:
    file.write(json.dumps(partition))  # use `json.loads` to do the reverse

with open('labels.dict', 'w') as file:
    file.write(json.dumps(labels))  # use `json.loads` to do the reverse

with open('class_id_indexes.dict', 'w') as file:
    file.write(json.dumps(class_id_indexes))  # use `json.loads` to do the reverse
