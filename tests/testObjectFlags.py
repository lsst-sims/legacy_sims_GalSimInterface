"""
Test code for ObjectFlags class.
"""
from collections import OrderedDict
import unittest
from lsst.sims.GalSimInterface import ObjectFlags

class ObjectFlagsTestCase(unittest.TestCase):
    """Test case class for ObjectFlags."""
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ObjectFlags(self):
        """Unit test for ObjectFlags class."""
        conditions = OrderedDict([(condition, index) for index, condition
                                  in enumerate('abcde')])
        flags = ObjectFlags(conditions=list(conditions.keys()))
        for condition, index in conditions.items():
            flags.set_flag(condition)
            self.assertEqual(2**index, flags.value)
            flags.unset_flag(condition)
            self.assertEqual(0, flags.value)

if __name__ == '__main__':
    unittest.main()
