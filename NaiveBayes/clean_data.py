import gzip
import tarfile
import os
from email.parser import Parser
import csv

path = '../../../dataset/enron/enron_mail_20150507.tar.gz'

enron_tar = tarfile.open(path, "r:gz")
mails = []
for member in enron_tar.getmembers():
    if member.isfile():
        f = enron_tar.extractfile(member)
        if f is not None:
            mailmap = {}
            data = f.read()
            mail = Parser().parsestr(data)

            mail_to = mail['to']
            if mail_to is not None:
                mailmap['to'] = mail_to.replace("\n", "").replace("\t", "").replace(" ", "")
            else:
                mailmap['to'] = ""

            mail_from = mail['from']
            if mail_from is not None:
                mailmap['from'] = mail_from.replace("\n", "").replace("\t", "").replace(" ", "")
            else:
                mailmap['from'] = ""

            mail_content = mail.get_payload()
            if mail_content is not None:
                mailmap['content'] = mail_content.
            else:
                mailmap['content'] = ""

            mail_subject = mail['subject']
            if mail_subject is not None:
                mailmap['subject'] = mail_subject
            else:
                mailmap['subject'] = ""
            mails.append(mailmap)
print(mails[0])
print(len(mails))

#write mails array to a csv
keys = mails[0].keys()
with open('cleaned_mail.csv', 'wb') as f:
    writer = csv.DictWriter(f, keys)
    writer.writeheader()
    writer.writerows(mails)



enron_tar.close()

