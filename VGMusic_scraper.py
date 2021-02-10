import requests
from bs4 import BeautifulSoup as bs
import os


root_url = 'https://www.vgmusic.com/'


page = requests.get(root_url).text
soup = bs(page, 'html.parser')

# links that lead to a './music/...' endpoint
links = [link.get('href') for link in soup.find_all('a') if link.get('href')[:7] == './music']



# `links` above gives us a flat list of the endpoints at which we
# can find all the midi files for a given console. We will save the download links
# in a dictionary, the keys of which are the gaming console and the values a list of links
# to download the files.


download_links = {}


def url_to_key(url):
	"""
	convert a url saved in `links` to a key
	in `download_links` dictionary
	"""
	return url.split('/')[-3] + ' ' + url.split('/')[-2]


scrape_tally = 0
download_tally = 0

if not os.path.exists("VGScrapes"):
	os.makedirs("VGScrapes")

for link in links:

	# set up scraping constants
	page_url = root_url + link[2:]
	
	print()
	print(f'Scraping page {links.index(link)+1} / {len(links)}, "{page_url}"')
	print()
	
	page_text = requests.get(page_url).text
	page_soup = bs(page_text, 'html.parser')

	# filter to just .mid file types
	midi_files = [page_url + file_link.get('href') for file_link in page_soup.find_all('a') 
		 		  if file_link and file_link.get('href') and file_link.get('href')[-4:] == '.mid']

	if midi_files:
		scrape_tally += len(midi_files)
		print(f"There are {len(midi_files)} .mid files at this endpoint")

		download_links[url_to_key(link)] = midi_files

		print('--------------------------')

		# create directory if none (will generally be the case on first use)
		dirname = url_to_key(link)
		
		if not os.path.exists('VGScrapes/' + dirname):
			os.makedirs('VGScrapes/' + dirname)

		print('scraping and downloading...')
		print()

		for index, file in enumerate(midi_files):

			if (index % 100) == 0 or index == len(midi_files)-1:
				print(f'{index} files downloaded so far.')
			
			try:
				r = requests.get(file, allow_redirects = True)
				try:
					open(f"VGScrapes/{dirname}/{file.split('/')[-1]}", 'wb').write(r.content)
				except:
					print(f"Failed to save file to desired directory")

			except:
				print(f'request to {file} failed')
		
		### Verify that/if number of files actually downloaded matches the number of filenames scraped	
		num_downloaded_files = len(os.listdir(os.path.expanduser(f'~/VGScrapes/{dirname}')))
		print(f'Successfully downloaded {num_downloaded_files} out of {len(midi_files)} scraped files')
		download_tally += num_downloaded_files
		print('############################')


	else:
		print("This endpoint has no midi files to offer")
		print()


### Print out total number of files
print(f"Finished downloading {download_tally} files from a total of {scrape_tally} scraped download links.")


