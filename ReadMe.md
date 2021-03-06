## Linear Algebra Module
### Katrina Wheelan
### (Spring 2018)

1. Files in this repository:
   - **README** - this file, an overview of the project
   - **LinAlg** - A module containing all classes and functions in a linear algebra module. This module can be used on its own in the command line, or within main.py.
   - **main.py** - A python script that creates a more user-friendly interface to use the LinAlg module.

2. Additional modules required by this software: none

3. Demonstrable accomplishments of this project: 
In this project, I created an algorithm to efficiently row reduce a matrix. This, in turn, we can use to solve systems of linear equations. Other functions in the interface include: row-reducing any matrix, finding the determinant, finding rational eigenvalues, and finding the inverse of a matrix. On its own, the Linear Algebra module contains several functions that could be useful in other programs. The module contains three classes: Matrix, Row, and ColVector (column vector). Each class contains several useful methods containing, but not limited to: addition (of Rows, ColVectors, and Matrices), scalar multiplication (of all three classes), dot product (between Rows or ColVectors), matrix multiplication, determinants, inversion, and eigenvalue calculations.


4. Documentation of use:
As an imported module, the LinAlg module can be used to do functions and calculations not possible within the normal Python framework.
In the interface, users can type a menu option and follow the directions to do the desired calculation.
Example run of main.py:

```
user$ python3 main.py
This is a linear algebra module.
Here is the menu:
Type 1 for: Solving a system of equations.
Type 2 for: Row-reducing a matrix.
Type 3 for: Finding an inverse of a matrix.
Type 4 for: Finding a determinant of a matrix.
Type 5 for: Finding rational eigenvalues of a matrix.
Type 'quit' to exit at any time.
Choice: 1
Enter equations one-by-one. They must by in the form:
          ax+by=c or ax + by = c; constants must be on the right side.
Type equation or type 'go' after inputting all equations. 2x + 3y =10
Type equation or type 'go' after inputting all equations. x +-y=1
Type equation or type 'go' after inputting all equations. go
Solution set: {'x': 2.6, 'y': 1.6}
Continue? [y/n] y
Here is the menu:
Type 1 for: Solving a system of equations.
Type 2 for: Row-reducing a matrix.
Type 3 for: Finding an inverse of a matrix.
Type 4 for: Finding a determinant of a matrix.
Type 5 for: Finding rational eigenvalues of a matrix.
Type 'quit' to exit at any time.
Choice: 2
How many rows are in the matrix? This must be an integer. 2
Enter each row as a list.
Row: [1,2]
Row: [3,4]
Row reduced matrix:
[1, 0]
[0, 1]
Continue? [y/n] n
```

5. Additional comments:
Make sure to input all text in the format the prompt specifies. (i.e. enter an integer when it says ‘This must be an integer’ and enter in the form ‘[1,2,3]’ when it asks for a list).




