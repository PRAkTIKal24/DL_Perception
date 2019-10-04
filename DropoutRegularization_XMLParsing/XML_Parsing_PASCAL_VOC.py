import csv
from os import listdir
from PIL import Image

#Read images 
fname = os.path.join(voc_path, 'VOC{}/JPEGImages/{}.jpg'.format(year,                                                                    image_id))
with open(fname, 'rb') as in_file:
    data = in_file.read()

#Using the RAM approach, (not creating temporary folder)
for file in listdir(~/VOCdevkit/VOC2012/Annotations):
  
  #Parsing data
  with open(file, "rb"):
    tree = ET.parse(file)
    root = tree.getroot()

    #List for storing the xml data in each file  
    list_with_all_boxes = []

    for boxes in root.iter('object'):

    	#List entries
        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        img = data.list_with_single_boxes
        list_with_all_boxes.append(list_with_single_boxes)

#Resize
for item in img:
	im = Image.open(img)
	imResize = im.resize((200,200), Image.ANTIALIAS)
    imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

#Write resized images into a binary file. The b in 'wb' describes the file type of Parsed_data.csv
with open('Parsed_data.csv', 'wb') as f:
	csvwriter = csv.writer(f, lineterminator = '\n')
	csvwriter.writerow(imResize)
