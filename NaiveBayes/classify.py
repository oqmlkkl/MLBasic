import csv
import sys
csv.field_size_limit(sys.maxsize)

with open('./cleaned_mail.csv', 'rb') as emails:
    reader = csv.DictReader(emails)
    for row in reader:
        print row['to']
