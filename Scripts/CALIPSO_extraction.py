# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 00:31:41 2021

@author: fafri
"""

import os
os.environ['PROJ_LIB'] = r"C:\Users\fafri\anaconda3\pkgs\proj4-6.1.1-hc2d0af5_1\Library\share"
import urllib
import fnmatch
import lxml.html


dirlist = ["https://xfr139.larc.nasa.gov/f0c1d3dd-9071-4d87-82c4-73e845fbb6bb/",
           "https://xfr139.larc.nasa.gov/7adf3193-3520-41d3-b87e-591212dacdc9/",
           "https://xfr139.larc.nasa.gov/0773bd02-b303-44d5-bd7c-78aa64d25c65/",
           "https://xfr139.larc.nasa.gov/feda055f-9da5-4fe6-8e09-de1fc780bbcc/",
           "https://xfr139.larc.nasa.gov/4962dbbc-dcd4-457d-9174-3dac0e5b1a00/",
           "https://xfr139.larc.nasa.gov/5ee8e267-9b2b-4e08-a07b-cad818ada998/",
           "https://xfr139.larc.nasa.gov/ef570a9e-a0fb-4912-997b-c7bd920ee298/",
           "https://xfr139.larc.nasa.gov/f08b2f52-6852-467a-a08a-743359320202/"]
# 03/19, 06/19, 09/19, 12/19, 03/20, 06/20, 09/20, 12/20

def url_list(url):
    urls = []
    connection = urllib.request.urlopen(url)
    dom =  lxml.html.fromstring(connection.read())
    for link in dom.xpath('//a/@href'):
        urls.append(link)
    return urls

#directory = 'https://xfr139.larc.nasa.gov/b32436ea-4373-4855-9bac-4a3a9eee57aa/'
#files = []

for idx in range(len(dirlist)):
    print(idx)
    urls = url_list(dirlist[idx])
    
    filetype = "*.hdf"
    file_list = [filename for filename in fnmatch.filter(urls, filetype)]
    #files.append(file_list)
        
    with open('file_03_19_12_20.dat', 'a') as text_file:
        for file in file_list:
            name = '{0}{1}'.format(dirlist[idx], file)
            text_file.write(name + '\n')



#'''
#IMPORT DATA: wget --load-cookies C:\Users\fafri\.urs_cookies --save-cookies C:\Users\fafri\.urs_cookies --auth-no-challenge=on --keep-session-cookies --user=timomaster --ask-password --header "Authorization: Bearer c3bc08e7ead1b953f28ee12773dd6a9519f94457c801a1d4799f63af9efd2e90" --content-disposition -i C:\Users\fafri\file_03_19_12_20.dat -P C:\Users\fafri\
#
#'''