import os

def main():
    print("Select an option to run:")
    print("1. Run main.py for reproducing results.")
    print("2. Run guifinal.py for reproducing results using GUI.")
    print("3. Run prediction.py for prediction on trained model.")
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == '1':
        os.system('python main.py')
    elif choice == '2':
        os.system('python guifinal.py')
    elif choice == '3':
        os.system('python prediction.py')
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()