import requests
from bs4 import BeautifulSoup as bs


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


tally = 0

for link in links[:1]:

	# redefine bs parameters
	page_url = root_url + link[2:]
	
	print()
	print(f'Scraping page {links.index(link)+1} / {len(links)}, "{page_url}"')
	print()
	
	page_text = requests.get(page_url).text
	page_soup = bs(page_text, 'html.parser')

	midi_files = [page_url + file_link.get('href') for file_link in page_soup.find_all('a') 
		 		  if file_link and file_link.get('href') and file_link.get('href')[-4:] == '.mid']

	if midi_files:
		tally += len(midi_files)
		print(f"There are {len(midi_files)} .mid files at this endpoint")

		download_links[url_to_key(link)] = midi_files

		print('--------------------------')

	else:
		print("This endpoint has no midi files to offer")
		print()


### Print out total number of files
print(f"Finished scraping a total of {tally} files")


### Download a song (NB this is just a test)
def download_test():
	fname = download_links['nintendo nes'][11]
	print(fname)
	r = requests.get(fname, allow_redirects = True)
	open(fname.split('/')[-1], 'wb').write(r.content)

