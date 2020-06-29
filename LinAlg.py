# (c) Katrina Wheelan 2018
"""A linear algebra module containing all classes and functions for this CS134 project.
These classes include: Matrix, Row, ColVector.
Functions include: identity, rowReduce, parseInput, parseOutput, solve, and getEqs."""

__all__ = ['Matrix','Row','ColVector', 'identity', 'rowReduce', 'parseInput', 'parseOutput', 'solve', 'getEqs']

from random import randint

class Row(object):
    """A class for a single row of a matrix"""
    __slots__ = ['_values', '_nentries']

    def __init__(self, valueList = [0]):
        assert type(valueList) is list
        self._values = valueList
        self._nentries = len(self._values)

    @property
    def nentries(self):
        """the number of entries in a row"""
        return self._nentries

    @property
    def values(self):
        """the values in a row"""
        return self._values

    def __str__(self):
        """A string representation of this row."""
        return str(self.values)

    def __repr__(self):
        """The eval-able representation of this row."""
        return "Row({})".format(self.values)

    def __eq__(self, other):
        """equality function of Row class."""
        return self.values == other.values

    def __iter__(self):
        """generator that iterates across entries"""
        for item in self.values:
            yield item

    def __getitem__(self, n):
        """returns self[n] = nth row of self"""
        if -(len(self.values)+1) <= n < len(self.values):
            return self.values[n]
        else:
            return None

    def __mul__(self, scalar):
        """returns a scalar multiple of a Row."""
        row = Row(list(self.values))
        for n in range(0,row.nentries):
            row._values[n] *= scalar
        return row

    def dot(self, other):
        """returns the dot product of two vectors"""
        assert self.nentries == other.nentries
        return sum([self[i]*other[i] for i in range(self.nentries)])

    def __add__(self, other):
        """returns the sum of two rows"""
        assert type(other) is type(self) and self.nentries == other.nentries
        return Row([ self[i] + other[i] for i in range(0,self.nentries) ])

    def __len__(self):
        """returns length of row"""
        return self.nentries

            
class ColVector(Row):
    """A class for a column vector. Most methods are inherited from the Row class."""
    __slots__ = ['_values','_nentries']

    def __init__(self, valueList):
        assert type(valueList) is list
        self._values = valueList
        self._nentries = len(valueList)

    def __str__(self):
        """string representation."""
        return "\n".join([str([entry]) for entry in self.values])

    def __repr__(self):
        """The eval-able representation of this column vector"""
        return "ColVector({})".format(self.values)


class Matrix(object):
    """A class for a matrix consisting of Row objects"""
    __slots__ = ['_values', '_colValues', '_nrows', '_ncols', '_augmented', '_variables']

    def __init__(self, rowList = [Row()], variables = [], augmented = 0):
        assert type(rowList) is list
        self._augmented = augmented #number of augmented columns
        self._nrows = len(rowList)
        self._ncols = 1
        self._variables = variables #a slot for a matrix representing equations
        self._values = []
        for row in rowList:
           # assert(type(row) is Row or type(row) is list)
            if type(row) is not Row:
                row = Row(row)
            self._values.append(row)
            if row.nentries > self._ncols:
                self._ncols = row.nentries
        for row in self._values:
            if row.nentries < self._ncols:
                row._values = row._values + [0]*(self._ncols - row.nentries)
                row._nentries = len(row.values)
        self._colValues = []
        for i in range(0, self._ncols):
            self._colValues.append(ColVector([self._values[row][i] for row in range(0,self._nrows)]))

    @property
    def nrows(self):
        """the number of rows in a matrix"""
        return self._nrows

    @property
    def ncols(self):
        """the number of columns in a matrix"""
        return self._ncols

    @property
    def values(self):
        """the values in a matrix"""
        return self._values

    @property
    def rows(self):
        """return matrix as list of rows."""
        return self._values

    @property
    def cols(self):
        """return matrix as list of column vectors."""
        return self._colValues

    def __str__(self):
        """A string representation of this matrix."""
        return '\n'.join([str(row) for row in self.values])

    def __repr__(self):
        """The eval-able representation of this matrix."""
        return "Matrix({})".format(self.values)

    def __eq__(self, other):
        """equality function of Matrix class"""
        return self.__repr__() == other.__repr__()

    def __add__(self, other):
        """addition function of Matrix class"""
        assert(self.nrows == other.nrows and self.ncols == other.ncols)
        return Matrix([self[i] + other[i] for i in range(0,self.nrows)])

    def __sub__(self, other):
        """subtraction function of Matrix class"""
        return self + other*-1

    def __mul__(self, other):
        """defines matrix multiplication"""
        if type(other) is Matrix:
            assert self.ncols == other.nrows
            return Matrix([[self[j].dot(other.cols[i]) for i in range(other.ncols)] for j in range(self.nrows)])
        if type(other) is int or float:
            return Matrix([row * other for row in self])

    def entry(self, row, col):
        """Returns entry at row and col"""
        return self.values[row].values[col]

    def __iter__(self):
        """generator that iterates across rows"""
        for row in self.values:
            yield row

    def __getitem__(self, n):
        """returns self[n] = nth row of self"""
        if -(len(self.values)+1) <= n < len(self.values):
            return self.values[n]
        else:
            return None

    def _updateCols(self):
        """method to update columns if rows change."""
        self._nrows = len(self.rows)
        self._ncols = len(self.rows[0])
        self._colValues = []
        for i in range(0, self._ncols):
            self._colValues.append(ColVector([self._values[row][i] for row in range(0,self._nrows)]))
 
    def _updateRows(self):
        """method to update rows if columns change."""
        self._values = []
        self._ncols = len(self.cols)
        self._nrows = len(self.cols[0]) 
        for i in range(0, self._nrows):
            self._values.append(Row([self._colValues[col][i] for col in range(0,self._ncols)]))       

    def swapRows(self, row1, row2):
        """swaps row1 and row2"""
        self._values[row1],self._values[row2] = self._values[row2],self._values[row1]
        self._updateCols()

    def combineRows(self, row1, scalar, row2):
        """sets row1 = row2 * scalar."""
        self._values[row1] = self._values[row1] + (self._values[row2] * scalar)
        self._updateCols()

    def augment(self, colVector):
        """augments matrix with a column vector."""
        assert type(colVector) is ColVector and len(colVector) == self.nrows 
        self._colValues.append(colVector)
        self._updateRows()
        self._augmented += 1

    def augmentMatrix(self, Matrix):
        """method to augment a matrix to an existing matrix"""
        for col in Matrix.cols:
            self.augment(col)

    def _smlMtx(self, row, col):
        """returns a matrix without a specified row and column.
        This method is used to find a determinant of a matrix."""
        rowless = Matrix(self._values[:row] + self._values[row+1:])
        output = Matrix()
        output._colValues = rowless._colValues[:col] + rowless._colValues[col+1:]
        output._updateRows()
        return output

    @property
    def det(self):
        """finds the determinant of a matrix recursively"""
        assert self.nrows == self.ncols
        if self.nrows == 2:
            return self[0][0]*self[1][1] - self[0][1]*self[1][0]
        return sum([self[0][i]*((-1)**i)*self._smlMtx(0,i).det for i in range(self.ncols)])

    @property
    def inv(self):
        """finds the inverse of a matrix by augmenting with identity and row reducing"""
        assert self.nrows == self.ncols and self.det
        inv = Matrix(self.rows)
        inv.augmentMatrix(identity(self.nrows))
        inv._colValues = rowReduce(inv).cols[self.nrows:]
        inv._updateRows()
        return inv

    @property
    def eigenvalues(self):
        """returns the rational eigenvalues of a matrix if it is invertible"""
        if self.det == 0:
            return("Not invertible; 0 is an eigenvalue, but cannot compute others.")
        n = abs(self.det)
        ratRts = [i for i in list(range(-n,0)) + list(range(1, n+1)) if n%i == 0]
        return [r for r in ratRts if (self - identity(self.nrows)*r).det ==0]

def identity(n = 1):
    """Returns identity matrix of size nxn."""
    return Matrix([ [0]*i + [1] for i in range(0,n) ])

def rowReduce(matrix):
    """Returns a row-reduced version of input matrix."""
    if matrix is None:
        return None
    assert type(matrix) is Matrix
    pivotRow = 0
    for pivot in range(0, matrix.ncols - matrix._augmented):
        newRow = pivotRow+1
        while pivotRow < matrix.nrows and matrix[pivotRow][pivot] == 0 and newRow < matrix.nrows:
            matrix.swapRows(pivotRow, newRow)
            newRow += 1
        if pivotRow < matrix.nrows and not matrix[pivotRow][pivot] == 0:
            matrix._values[pivotRow] = matrix[pivotRow] * (1/matrix[pivotRow][pivot])
            for row in range(0,matrix.nrows):
                matrix._values[row] = matrix[row] + matrix[pivotRow]*(-(row!=pivotRow)*matrix[row][pivot])
            pivotRow += 1               
    for row in matrix._values:
        for entry in range(0,matrix.ncols):
            if row._values[entry]%1 == 0:
                row.values[entry] = int(row.values[entry])
            else:
                row.values[entry] = round(row.values[entry], 3)
    matrix._updateCols()
    return matrix

def getEqs():
    """asks for input of equations."""
    eq = ''
    eqList = []
    print("""Enter equations one-by-one. They must by in the form:
          ax+by=c or ax + by = c; constants must be on the right side.""")
    while eq not in ['go','quit']:
        eq = input("Type equation or type 'go' after inputting all equations. ")
        if not eq == 'go':
            eqList.append(eq)
    return eqList

def parseInput(eqList):
    """parses input."""
    if not eqList:
        return None
    constants = ColVector([int(equation.split('=')[1].strip()) for equation in eqList])
    leftSides = [equation.split('=')[0] for equation in eqList]
    variables = list({term.strip()[-1] for equation in leftSides for term in equation.split('+')})
    rowDicts = []
    for equation in leftSides:
        d = dict()
        for term in equation.split('+'):
            if not term.strip()[:-1]:
                d[term.strip()[-1]] = 1
            elif term.strip()[:-1] == '-':
                d[term.strip()[-1]] = -1
            else:
                d[term.strip()[-1]] = int(term.strip()[:-1])
                
        rowDicts.append(d)
    cols = []
    for var in variables:
        column = [dictionary.get(var,0) for dictionary in rowDicts]
        cols.append(ColVector(column))
    matrix = Matrix(variables=variables)
    matrix._colValues = cols
    matrix._updateRows()
    matrix.augment(constants)
    return(matrix)

def parseOutput(matrix):
    """parses row-reduced matrix
    attempting to write for free variables"""
    if matrix is None:
        return None
    assert(len(matrix._variables) == matrix.ncols-1 and matrix._augmented)
    varDict = dict()
    for row in range(matrix.nrows):
        if sum(matrix[row].values[:-1]) == 0 and matrix[row].values[-1] != 0:
            return("Inconsistent system.")
        for col in range(matrix.ncols-1):
            if not matrix[row][col] == 0:
                if sum(matrix[row].values) == matrix[row][col] + matrix[row][-1]:
                    varDict[matrix._variables[col]] = matrix[row][-1]
                else:
                    l = []
                    for i in range(col+1, matrix.ncols-1):
                        if matrix[row][i] != 0:
                            l.append(str(-matrix[row][i])+matrix._variables[i])
                    if not l:
                        varDict[matrix._variables[col]] = "free"
                    else:
                        varDict[matrix._variables[col]] = str(matrix[row][-1]) +"+"+ "+".join(l)
    return varDict

def solve():
    """solves an inputted system of equations."""
    return parseOutput(rowReduce(parseInput(getEqs())))

def _createSys(nVars = 3):
    """creates a random system of equations.
    This is mostly to check that the rowReduction method works properly"""
    from random import randint
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    varAmts = dict()
    for var in range(nVars):
        varAmts[alphabet[var]] = randint(-9,10)
    eqs = []
    for var in range(nVars):
        terms = [(randint(-9,10), var) for var in varAmts]
        rSide = str(sum([term[0]*varAmts[term[1]] for term in terms]))
        eq = "+".join([(str(term[0])+term[1]) for term in terms])+' = '+rSide
        eqs.append(eq)
    return(eqs)

def _check(eqList):
    """simply a function to check row reduction using createSys function."""
    return parseOutput(rowReduce(parseInput(eqList)))

