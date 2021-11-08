import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


def SendMail(ImgFileName):
    img_data = open(ImgFileName, 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = 'Security Alert'
    msg['From'] = 'shubhi.coolio21@gmail.com.cc'
    msg['To'] = 'ananyagupta.bt19cse@pec.edu.in.cc'

    text = MIMEText("Alert Message. Review the frame and contact authorities")
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    msg.attach(image)

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login("shubhi.coolio21@gmail.com", "pleasefacebook")
    s.sendmail("shubhi.coolio21@gmail.com", "ananyagupta.bt19cse@pec.edu.in", msg.as_string())
    s.quit()