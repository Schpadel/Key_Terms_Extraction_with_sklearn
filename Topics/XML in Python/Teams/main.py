#  write your code here 

from lxml import etree

tree = etree.parse("data/dataset/input.txt")

root = tree.getroot()

for element in root[0]:
    print(element.get("name"), end=" ")
