"""Root conftest.py — ensures repo root is on sys.path for pytest."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))