import os
from bs4 import BeautifulSoup


main_location = r'D:\Datasets\IsItCorrect\reddit_french_conversation_crawled'
main_file_location = os.path.join(main_location, 'crawl_1.xml')
save_file_location = os.path.join(main_location, 'parsed_text.txt')

start_token = b"<START>"
end_token = b"<END>"

with open(main_file_location, 'r', encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'html.parser')

with open(save_file_location, 'wb') as filesave:
    for i in soup.find_all('s'):
        blocks = i.text.strip()
        blocks = blocks.split("\n")
        for block in blocks:
            filesave.write(start_token)
            filesave.write(bytes(block, 'utf-8'))
            filesave.write(end_token)
