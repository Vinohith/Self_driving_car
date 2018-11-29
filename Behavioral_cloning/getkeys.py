# adapted from https://github.com/recantha/EduKit3-RC-Keyboard/blob/master/rc_keyboard.py
 
import sys, termios, tty, os, time
 
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
 
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
 
# button_delay = 0.2
characters = []
while True:
	char = getch()
	if char == 'w':
		print('Forward')
	elif char == 'a':
		print('Left')
	elif char == 'd':
		print('Right')
	elif char == 'p':
		break
	else:
		print('Invalid')

	characters.append(char)
print(characters)