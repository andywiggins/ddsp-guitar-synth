# Author: Andy Wiggins <awiggins@drexel.edu>
# Functions for simple user interfacing in colab

import getpass

def list_selection_menu(L):
    """
    Display a menu of options from a list. Have the user type an index to select an option. Return the corresponding list object. 

    Parameters
    ----------
    L : list of objects
        List of options for the user to select from

    Returns
    ----------
    sel_obj : object
        Returns the selected object
    """
    print("Type a number to select an option:")
    for i, obj in enumerate(L):
        print(f"\t{i})\t{obj}")
    while(True):
        usr_input = input("Selection: ")
        try:
            sel_obj = L[int(usr_input)]
        except:
            print("\tInvalid index. Try again.")
        else:
            print(sel_obj)
            return sel_obj # exit the loop, nothing went wrong
    

    


		
	


	
	




    

    
        
        
