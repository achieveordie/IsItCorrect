from bs4 import BeautifulSoup
import requests
import os
import PyPDF2

start_token = "<START>"
end_token = "<END>"

french_data_save_location = r'D:\Datasets\IsItCorrect\stories.txt'
french_novel_save_location = r'D:\Datasets\IsItCorrect\novel.txt'

url_list = [
    "https://www.thefablecottage.com/french/blanche-neige",
        "https://www.thefablecottage.com/french/tracassin",
        "https://www.thefablecottage.com/french/raiponce",
        "https://www.thefablecottage.com/french/les-habits-neufs-de-lempereur",
        "https://www.thefablecottage.com/french/cendrillon",
        "https://www.thefablecottage.com/french/jack-et-les-haricots-magiques",
        "https://www.thefablecottage.com/french/hansel-et-gretel",
        "https://www.thefablecottage.com/french/les-trois-boucs-bourrus",
        "https://www.thefablecottage.com/french/petit-poulet",
        "https://www.thefablecottage.com/french/loiseau-et-la-baleine"
]

for url in url_list:
    html_data = requests.get(url).text
    soup = BeautifulSoup(html_data, "html.parser")
    paras = soup.find_all('p')
    text_doc = ""
    for i in range(len(paras)):
        if i%2 == 0 and i < len(paras)-3:
            text_doc += start_token
            text_doc += paras[i].text
            text_doc += end_token
    with open(french_data_save_location, 'a', encoding='utf8') as file:
        file.write(text_doc)


pdf_location = r'C:\Users\Sagar Mishra\Downloads'
file_locations = [os.path.join(pdf_location, 'La Grande Aventure.pdf'),
         os.path.join(pdf_location, 'La société.pdf')]

for file_location in file_locations:
    with open(file_location, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        text = ""
        num_pages = pdf_reader.numPages
        for i in range(num_pages):
            pageObj = pdf_reader.getPage(i)
            text += pageObj.extractText()
    with open(french_novel_save_location, 'a', encoding='utf8') as file:
        file.write(start_token)
        for t in text:
            if t != '.':
                file.write(t)
            else:
                file.write(end_token)
                file.write(start_token)