"""
Unit testing for utilities4cotagging
"""
import pytest
import os
from utilities4cotagging import *

script_path = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(script_path, 'testfiles')

@pytest.mark.parametrize("test_input,expected", [
    (os.path.join(test_folder, 'file_with_10_lines.txt'), 10),
    (os.path.join(test_folder, 'file_with_20_lines.txt'), 20)])
def test_mapcount(test_input,expected):
    assert mapcount(test_input) == expected


@pytest.mark.parametrize("test_input,expected",
                         [(os.path.join(test_folder,'test_log'), 197)])
def test_read_log(test_input,expected):
    profs = read_log(test_input)
    assert len(profs) == expected


@pytest.mark.parametrize("test_input,expected",
                         [('ls test_utilities4cotagging.py',
                           (b'test_utilities4cotagging.py\n', b''))])
def test_executeLine(test_input,expected):
    assert executeLine(test_input) == expected


