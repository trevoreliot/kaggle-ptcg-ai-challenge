import argparse
import sys
import main

# Mock args to run the evaluate loop for 1 episode
sys.argv = ["main.py", "--mode", "evaluate", "--episodes", "1", "--p2-type", "rules", "--p2-agent", "iono_s_deck"]
main.main()
