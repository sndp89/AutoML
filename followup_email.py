"""
    followup_email.py
    
    The code here is used to send email to users.  
"""

from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import os,string,sys

def emailer(nt_id, email_id, mdl_prefix, filenames, message):
    
    SERVER = 'rdkmail.stb.r53.xcal.tv'
    FROM = ["ericathebot@comcast.com"]
    TO = [email_id] # must be a list
    SUBJECT = "Automated Model Building for '" + mdl_prefix + "' model. Output files"
    TEXT = "Hello, \n \n " + message + "\n \n Regards, \n EBI Data Science Automation Team"

    msg = MIMEMultipart()

    msg['From'] = ", ".join(FROM)
    msg['To'] = ", ".join(TO)
    msg['Subject'] = SUBJECT

    body = TEXT

    msg.attach(MIMEText(body, 'plain'))
    
    try:
        for filename in filenames:
            name = mdl_prefix + '.zip'
            attachment = open(filename, "rb")
            part = MIMEBase('application', 'octet-stream')
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', "attachment; filename= %s" % name)
            msg.attach(part)
    except:
        pass

    message = msg.as_string()

    server = smtplib.SMTP(SERVER,25)
    server.sendmail(FROM, TO, message)
    server.quit()
    
    print('Email sent successfully!')