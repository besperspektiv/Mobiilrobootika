
from win32gui import GetWindowText, GetForegroundWindow
import pyautogui
import time
import random
import string
import secrets
import random

window = GetWindowText(GetForegroundWindow())
flag = 0
Browser_flag = 0


letters = string.ascii_letters
digits = string.digits
special_chars = string.punctuation
alphabet = letters + digits + special_chars


def Upper_Lower_string(length):  # define the function and pass the length as argument
    # Print the string in Lowercase
    result = ''.join(
        (random.choice(string.ascii_lowercase) for x in range(length)))  # run loop until the define length

    # Print the string in Uppercase
    result1 = ''.join(
        (random.choice(string.ascii_uppercase) for x in range(length)))  # run the loop until the define length
    final_result = result + result1
    print(final_result)
    return final_result

def Password(length):
    pwd = ''
    for i in range(length):
        pwd += ''.join(secrets.choice(alphabet))
    pwd = pwd + str(random.randint(0, 100))
    print(pwd)
    return pwd


while True:
    current_window = GetWindowText(GetForegroundWindow())
    if(current_window != window):
        flag = 0
        window = current_window
        time.sleep(0.2)
    if(window == "Введите пароль для G:\logo" and flag == 0):
        flag = 1
        pyautogui.write('123456789')

        pyautogui.press('enter')

    if "Browser" in window and Browser_flag == 0:
        pass
    length = random.randint(8, 15)
    username = Upper_Lower_string(length)
    length = random.randint(15, 30)
    password = Password(length)

    pyautogui.write(username)
    pyautogui.press('tab')
    pyautogui.write(password)
    pyautogui.press('tab')
    pyautogui.write(password)
    Browser_flag = 1







