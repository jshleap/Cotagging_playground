"""
Unit testing for qtraitsimulation
"""
import pytest
from qtraitsimulation_old import *

# Constants for tests
script_path = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(script_path, 'testfiles')
bed1 = os.path.join(test_folder, 'toy_bed')
bed2 = os.path.join(test_folder, 'toy_bed2')


# Tests
@pytest.mark.parametrize("bed,h2,ncausal,,normed,uni,bfile2,ceff", [
    ('toy_trueprs_0.5_5_0.01_norm', 0.5, 5, None, True, None, None, False)
])
def test_true_prs():

