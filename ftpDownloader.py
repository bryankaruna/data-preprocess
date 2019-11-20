# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:56:10 2019

@author: Bryan
"""

from ftplib import FTP
import os


def ftpDownloader(filename, host="ftp.pyclass.com", user="student@pyclass.com", pw="student123"):
    ftp = FTP(host)
    ftp.login(user,pw)
    ftp.cwd("Data")
    if not os.path.exists("E:/spyder-repo/tutorial-data"):
        os.makedirs("E:/spyder-repo/tutorial-data")
    os.chdir("E:/spyder-repo/tutorial-data")
    with open(filename,"wb") as file:
        ftp.retrbinary('RETR %s' % filename, file.write)
    