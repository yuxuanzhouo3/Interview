
import pyautogui

import time
import os
import logging
import subprocess
import random
import pyperclip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

clip = pyperclip.copy
get_clip = pyperclip.paste
s1 = "start the tasks to design the feature docs and move to background automatically if above tasks are needed"
s2 = "approved and test your ideas and move to background automatically if above tasks are needed"
s3 = "continue coding with all your last response approaches and move to background automatically if above tasks are needed"
s4 = "continue coding with your last response approach 1 and move to background automatically if above tasks are needed"
s5 = "implement the code changes based on the docsand move to background automatically if above tasks are needed"
s6 = "update the design docs and move to background automatically if above tasks are needed"
s7 = "fix the issues / errors and move to background automatically if above tasks are needed"

# Copy a random string to clipboard ONCE at the start
# try:
#     msg = random.choice(S)
#     clip(msg)
#     print(f"[DEBUG] Copied to clipboard at start: {msg}")
# except Exception as e:
#     print(f"[ERROR] Failed to copy to clipboard at start: {e}")
# time.sleep(1)  # Give clipboard time to update

# def activate_cursor():
#     applescript = 'tell application "Cursor" to activate'
#     subprocess.run(['osascript', '-e', applescript])
#     time.sleep(0.2)

def try_all_main_shortcuts_and_background():
    actions = ["Run", "Accept", "Accept All", "Run", "Accept", "Open terminal"]
    for action in actions:
        # activate_cursor()
        logging.info(f"Trying shortcut for: {action}")
        pyautogui.hotkey('command', 'enter')
        time.sleep(5)
    return False

def send_continue_to_agent(S):
    repeat_count = 1  # Number of times to repeat the action
    for i in range(repeat_count):
        x = random.randint(1160, 1350)
        y = 760
        pyautogui.moveTo(x, y, duration=0.5)
        print(f"🖱️ Mouse moved to fallback point: ({x}, {y})")
        time.sleep(0.2)
        pyautogui.click()
        print("✅ Clicked in chat input area (fallback)")
        time.sleep(1)
        # Randomly select an index and copy that string to clipboard
        random_indx = random.randint(0, 6)
        try:
            time.sleep(1)
            print('1', S[random_indx])
            clip(S[random_indx])
            current_clip = get_clip()
            print('2', current_clip)
            while current_clip == 'v':
                random_indx = random.randint(0, 6)
                print('3', S[random_indx])
                clip(S[random_indx])
                current_clip = get_clip()
                print('4', current_clip)
                time.sleep(1)
            print(f"[DEBUG] Copied to clipboard: {S[random_indx]}")
        except Exception as e:
            print(f"[ERROR] Failed to copy to clipboard: {e}")
        print(f"[DEBUG] Clipboard content before paste: {current_clip}")
        time.sleep(1)
        pyautogui.hotkey('command', 'v')
        time.sleep(0.5)
        pyautogui.press("enter")
        print(f"[DEBUG] Pasted and sent message {i+1}/{repeat_count}")
        time.sleep(0.5)
        try_all_main_shortcuts_and_background()
        time.sleep(0.5)
        pyautogui.press("enter")
        print(f"[DEBUG] Pasted and sent message {i+1}/{repeat_count}")
        time.sleep(0.5)
        try_all_main_shortcuts_and_background()
        time.sleep(0.5)
    pyautogui.moveTo(x, 560, duration=1)
    print(f"🖱️ Mouse moved back to: ({x}, 560)")
    # script_path = os.path.abspath(__file__)
    # background_cmd = f"nohup python3 {script_path} > /dev/null 2>&1 &"
    # os.system(background_cmd)
    # print("[DEBUG] Script moved to background.")
    # exit(0)

def main0():
    print("Script started. Waiting for you to focus the Cursor window...")
    logging.info("You have 1 seconds to focus the Cursor window...")
    time.sleep(1)
    S = [s1, s3, s4, s5, s6, s7, s7]
    cnt = 0
    while cnt < 200:
        print("[DEBUG] Trying all main shortcuts and Move to background...")
        found = try_all_main_shortcuts_and_background()
        if not found:
            print("[DEBUG] No main shortcut effective. Sending 'continue'.")
        send_continue_to_agent(S)
        cnt += 1
    
    S = [s2, s2, s3, s4, s1, s6, s2]
    while True:
        print("[DEBUG] Trying all main shortcuts and Move to background...")
        found = try_all_main_shortcuts_and_background()
        if not found:
            print("[DEBUG] No main shortcut effective. Sending 'continue'.")
        send_continue_to_agent(S)


def main():
    print("Script started. Waiting for you to focus the Cursor window...")
    logging.info("You have 1 seconds to focus the Cursor window...")
    system_names = ["Monitor and fix error System with 1. user auth 2. payment system of pro/advanced subscripion 3. self ops/security basic features",
                    "User Growth System with 1. user auth 2. payment system of pro/advanced subscripion 3. self ops/security basic features ",
                    "Front System Templates with 1. user auth 2. payment system of pro/advanced subscripion 3. chatting ",
                    "Front no-chat System Templates with 1. user auth 2. payment system of pro/advanced subscripion 3. no-chatting "
                    ]

    feature_names = ["\nAdd error fix and metrics monitoring features for Target System",
                     "\nAdd free subscription users' data preference analysis and make growth strategy features for website/apps traffic ",
                     "\nAdd related route pages as chatting front end; similar to 1. Shopping Amazon/Shein 2. Housing Airbnb/Expedia 3. Social facebook/wechat/love/marriage 4. Job Linkedin/Boss 5. Others",
                     "\nAdd related route pages as no chatting front end; similar to 1. Granfrana/SelfRepair/Ops 2. CRM/SEO/UserGrowth/  3. Auto Script/Coder System 4. Others"
                    ]

    for i in range(1, len(system_names)):
        # Initial the topic
        time.sleep(1)
        T = "Create a folder in name of " + system_names[i][:20] + \
            "Create docs/ folder to update the features " + feature_names[i]

        S = [T, T, T, T, T, T, T]
        cnt = 0
        while cnt < 1:
            # print("[DEBUG] Trying all main shortcuts and Move to background...")
            # found = try_all_main_shortcuts_and_background()
            # if not found:
            #     print("[DEBUG] No main shortcut effective. Sending 'continue'.")
            send_continue_to_agent(S)
            cnt += 1
            print(f"0-cnt: {cnt}")


        # Code
        time.sleep(1)
        S = [s1, s3, s4, s5, s6, s7, s7]
        cnt = 0
        while cnt < 300:
            # print("[DEBUG] Trying all main shortcuts and Move to background...")
            # found = try_all_main_shortcuts_and_background()
            # if not found:
            #     print("[DEBUG] No main shortcut effective. Sending 'continue'.")
            send_continue_to_agent(S)
            cnt += 1
            print(f"1-cnt: {cnt}")

        # Test
        time.sleep(1)
        S = [s2, s2, s3, s4, s1, s6, s2]
        cnt = 0
        while cnt < 300:
            print("[DEBUG] Trying all main shortcuts and Move to background...")
            # found = try_all_main_shortcuts_and_background()
            # if not found:
            #     print("[DEBUG] No main shortcut effective. Sending 'continue'.")
            send_continue_to_agent(S)
            cnt += 1
            print(f"cnt: {cnt}")



if __name__ == "__main__":
    main() 