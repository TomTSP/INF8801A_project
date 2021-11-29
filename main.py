# coding=utf-8
import sys
import tools
style = sys.argv[1]
style_mask = sys.argv[2]
style_lm = sys.argv[3]
ex = sys.argv[4]
ex_mask = sys.argv[5]
ex_lm = sys.argv[6]

tools.style_transfer(style, style_mask, style_lm, ex, ex_mask, ex_lm)