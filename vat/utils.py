
def debug_text(text):
    with open("debug.out", "a") as f:
        f.write(text + "\n")