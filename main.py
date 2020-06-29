#(c) Katrina Wheelan 2018

"""This program runs the interactive part of the project."""

from LinAlg import *

def enterMatrix(square = False):
    """function to enter a user-inputted matrix."""
    nrows = 'False'
    if square:
        print('This matrix must be square (number of rows must equal number of entries).')
    while nrows not in [0,'quit'] and type(eval(nrows)) is not int:
        nrows = input("How many rows are in the matrix? This must be an integer. ")
    if nrows == 'quit':
        quit()
    print("Enter each row as a list in the form [1,2,3].")
    rows = [eval(input("Row: ")) for row in range(eval(nrows))]
    for row in rows:
        if not type(row) is list:
            print("You must enter rows as *lists*.")
            enterMatrix()
        if square and len(row) != int(nrows):
            print("The number of rows must equal number of columns.")
            enterMatrix(square=True)
    return Matrix(rows)

print("This is a linear algebra module.")

response = ''
while response != "quit":
    print("Here is the menu:")
    print("Type 1 for: Solving a system of equations.")
    print("Type 2 for: Row-reducing a matrix.")
    print("Type 3 for: Finding an inverse of a matrix.")
    print("Type 4 for: Finding a determinant of a matrix.")
    print("Type 5 for: Finding rational eigenvalues of a matrix.")
    print("Type 'quit' to exit at any time.")
    response = ''
    while response not in ['1','2','3','4','5'] and not response == 'quit':
        response = input("Choice: ")
    if response == '1':
        print('Solution set: {}'.format(solve()))
    if response == '2':
        print("Row reduced matrix:\n{}".format(rowReduce(enterMatrix())))
    if response == '3':
        print("Inverse: \n{}".format(enterMatrix(square=True).inv))
    if response == '4':
        print("Determinant: {}".format(enterMatrix(square=True).det))
    if response == '5':
        print("Rational eigenvalues: {}".format(enterMatrix(square=True).eigenvalues))
    if response == 'quit':
        quit()
    cont = input("Continue? [y/n] ")
    if cont == 'n':
        response = 'quit'   
    
    


