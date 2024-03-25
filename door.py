from twilio.rest import Client

account_sid = 'ACceade456cb4bd327e9944d06365436c8'
auth_token = '3c6af0a79acbd83576e3527c59ca1aa3'
client = Client(account_sid, auth_token)

def sendSms():
    message = client.messages.create(
    from_='+14433414337',
    body='Alert : Unknown Detected !!',
    to='+918779823783'
    )

    print(message.sid)