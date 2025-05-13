
import pandas as pd
import numpy as np
import pytest
from spyc.control_rules import _rules_func

data = np.random.randn(100)
index = pd.date_range(start='2023-01-01', periods=100, freq='D')
df = pd.DataFrame()
df.index = index
df['target'] = data
    
cl = df['target'].mean()
ucl = cl + 3 * df['target'].std()
lcl = cl - 3 * df['target'].std()

df['cl'] = cl
df['lcl'] = lcl
df['ucl'] = ucl

def test_rule_1_violation():

    series = df['target']
    cl = df['cl'] 
    lcl = df['lcl'] 
    ucl = df['ucl'] 
    
    rules_dict = {"Rule 1": True, "Rule 2": False, "Rule 3": False, "Rule 4": False, "Rule 5": False}
    
    result = _rules_func(series, cl, lcl, ucl, rules_dict)
    
    assert "Rule 1 violation" in result

def test_rule_2_violation():

    series = df['target']
    cl = df['cl'] 
    lcl = df['lcl'] 
    ucl = df['ucl'] 
    
    rules_dict = {"Rule 1": False, "Rule 2": True, "Rule 3": False, "Rule 4": False, "Rule 5": False}
    
    result = _rules_func(series, cl, lcl, ucl, rules_dict)
    
    assert "Rule 2 violation" in result

def test_rule_3_violation():

    series = df['target']
    cl = df['cl'] 
    lcl = df['lcl'] 
    ucl = df['ucl'] 
    
    rules_dict = {"Rule 1": False, "Rule 2": False, "Rule 3": True, "Rule 4": False, "Rule 5": False}
    
    result = _rules_func(series, cl, lcl, ucl, rules_dict)
    
    assert "Rule 3 violation" in result

def test_rule_4_violation():

    series = df['target']
    cl = df['cl'] 
    lcl = df['lcl'] 
    ucl = df['ucl'] 
    
    rules_dict = {"Rule 1": False, "Rule 2": False, "Rule 3": False, "Rule 4": True, "Rule 5": False}
    
    result = _rules_func(series, cl, lcl, ucl, rules_dict)
    
    assert "Rule 4 violation" in result

def test_rule_5_violation():

    series = df['target']
    cl = df['cl'] 
    lcl = df['lcl'] 
    ucl = df['ucl'] 
    
    rules_dict = {"Rule 1": False, "Rule 2": False, "Rule 3": False, "Rule 4": False, "Rule 5": True}
    
    result = _rules_func(series, cl, lcl, ucl, rules_dict)
    
    assert "Rule 5 violation" in result
