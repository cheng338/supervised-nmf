"""Bounded import-state ownership; no numerical import or subprocess."""
import os
import unittest
from unittest.mock import patch

from tests._thread_import_state import THREAD_ENVIRONMENT, preserve_thread_environment


class ThreadImportStateTests(unittest.TestCase):
    def test_present_caps_are_restored(self):
        values = {key: '1' for key in THREAD_ENVIRONMENT}
        values.update(OMP_THREAD_LIMIT='8', MKL_DYNAMIC='FALSE')
        with patch.dict(os.environ, values):
            with preserve_thread_environment():
                for key in THREAD_ENVIRONMENT:
                    os.environ[key] = '1'
            self.assertEqual({key: os.environ[key] for key in THREAD_ENVIRONMENT}, values)

    def test_missing_caps_are_missing_again(self):
        with patch.dict(os.environ):
            for key in THREAD_ENVIRONMENT:
                os.environ.pop(key, None)
            with preserve_thread_environment():
                for key in THREAD_ENVIRONMENT:
                    os.environ[key] = '1'
            self.assertTrue(all(key not in os.environ for key in THREAD_ENVIRONMENT))

    def test_import_error_keeps_exception_and_restores_mixed_state(self):
        error = RuntimeError('import sentinel')
        with patch.dict(os.environ, OMP_THREAD_LIMIT='8'):
            os.environ.pop('VECLIB_MAXIMUM_THREADS', None)
            with self.assertRaises(RuntimeError) as observed:
                with preserve_thread_environment():
                    os.environ['OMP_THREAD_LIMIT'] = '1'
                    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
                    raise error
            self.assertIs(observed.exception, error)
            self.assertEqual(os.environ['OMP_THREAD_LIMIT'], '8')
            self.assertNotIn('VECLIB_MAXIMUM_THREADS', os.environ)

    def test_nested_contexts_restore_their_own_boundary_only(self):
        with patch.dict(os.environ, OMP_THREAD_LIMIT='8'):
            with preserve_thread_environment():
                os.environ['OMP_THREAD_LIMIT'] = '2'
                with preserve_thread_environment():
                    os.environ['OMP_THREAD_LIMIT'] = '1'
                    os.environ['SNMF_IMPORT_TEST_SENTINEL'] = 'retained'
                self.assertEqual(os.environ['OMP_THREAD_LIMIT'], '2')
            self.assertEqual(os.environ['OMP_THREAD_LIMIT'], '8')
            self.assertEqual(os.environ['SNMF_IMPORT_TEST_SENTINEL'], 'retained')


if __name__ == '__main__':
    unittest.main()
