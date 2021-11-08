import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


def SendMail(ImgFileName):
    img_data = open(ImgFileName, 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = 'Security Alert'
    msg['From'] = 'insert_sender_email.cc'
    msg['To'] = 'insert_receiver_email.cc'

    text = MIMEText("Alert Message. Review the frame and contact authorities")
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    msg.attach(image)

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login("insert_sender_email", "insert_sender_password")
    s.sendmail("insert_sender_email", "insert_receiver_email", msg.as_string())
    s.quit()
