#!/usr/bin/env python
# coding:utf8

"""
this module reads strings.csv, which contains all
the strings, and lets the main app use it
"""

import sys
import csv
import os
from flask import Markup
import configparser

config = configparser.RawConfigParser()
path = '../hseling_api_diachrony_webvectors/hseling_api_diachrony_webvectors/webvectors.cfg'
assert os.path.isfile(path), "Current path: {}".format(os.getcwd())
config.read(path)

root = config.get('Files and directories', 'root')
l10nfile = config.get('Files and directories', 'l10n')

# open the strings database:
csvfile = open("../hseling_lib_diachrony_webvectors/hseling_lib_diachrony_webvectors/" + l10nfile, 'rU')
acrobat = csv.reader(csvfile, dialect='excel', delimiter=',')

# initialize a dictionary for each language:
language_dicts = {}
langnames = config.get('Languages', 'interface_languages').split(',')
header = next(acrobat)
included_columns = []
for langname in langnames:
    language_dicts[langname] = {}
    included_columns.append(header.index(langname))

# read the csvfile, populate language_dicts:
for row in acrobat:
    for i in included_columns:  # range(1, len(row)):
        # Markup() is used to prevent autoescaping in templates
        if sys.version_info[0] < 3:
            language_dicts[header[i]][row[0]] = Markup(row[i].decode('utf-8'))
        else:
            language_dicts[header[i]][row[0]] = Markup(row[i])
