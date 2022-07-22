import os 

def main():
    
    folder = "fotky"
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"{str(count)}.jpeg"
        src = f"{folder}/{filename}"
        dst = f"{folder}/{dst}"

        os.rename(src, dst)

if __name__ == '__main__':
    main() 