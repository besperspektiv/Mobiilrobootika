import win32con
import win32gui
import win32ui
from PIL import Image
import numpy as  np

def list_window_names():
    """ Prints list of windows.  """

    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            print(hex(hwnd), win32gui.GetWindowText(hwnd))

    win32gui.EnumWindows(winEnumHandler, None)




def crop_image_from_window(window_name):
    hwnd_target = win32gui.FindWindow(None, window_name)
    """Crop window image"""
    try:
        left, top, right, bot = win32gui.GetWindowRect(hwnd_target)
        w = right - left
        h = bot - top

        win32gui.SetForegroundWindow(hwnd_target)

        hdesktop = win32gui.GetDesktopWindow()
        hwndDC = win32gui.GetWindowDC(hdesktop)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

        saveDC.SelectObject(saveBitMap)

        result = saveDC.BitBlt((0, 0), (w, h), mfcDC, (left, top), win32con.SRCCOPY)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hdesktop, hwndDC)

        if result is None:
            open_cv_image = np.array(im)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            return open_cv_image
    except:
        img = np.zeros((300, 512, 3), np.uint8)
        print("Select right window!")
        return img
