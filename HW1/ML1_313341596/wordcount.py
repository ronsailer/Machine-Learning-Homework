# Copyright 2010 Google Inc.
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""Wordcount exercise

The main() below is already defined and complete. It calls print_words()
and print_top() functions which you write.

1. Implement a print_words(filename) function that counts
how often each word appears in the text and prints:
word1 count1
word2 count2
...

Print the above list in order sorted by word (python will sort punctuation to
come before letters -- that's fine). Store all the words as lowercase,
so 'The' and 'the' count as the same word.

2. Implement a print_top(filename) which is similar
to print_words() but which prints just the top 5 most common words sorted
so the most common word is first, then the next most common, and so on.

Use str.split() (no arguments) to split on all whitespace.

Workflow: don't build the whole program at once. Get it to an intermediate
milestone and print your data structure and sys.exit(0).
When that's working, try for the next milestone.

Optional: define a helper function to avoid code duplication inside
print_words() and print_top().

"""

import sys
from collections import Counter

def get_counter(filename):
    with open(filename, 'r') as f:
        return Counter(f.read().lower().split())


def print_words(filename):
    c = get_counter(filename)
    for word,count in sorted(c.most_common()):
        print word,count

def print_top(filename):
    c = get_counter(filename)
    for word, count in c.most_common(5):
        print word,count

###

# This basic main function is provided and
# calls the print_words() and print_top() functions which you must define.
def main():
  print 'print_words start'
  print_words('wordcount_input.txt')
  print 'print_words end'
  print 'print_top start'
  print_top('wordcount_input.txt')
  print 'print_top end'

if __name__ == '__main__':
  main()
