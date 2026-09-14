.PHONY: test test-cpp test-python sprt verify cg roundrobin opening-book book-inspect clean

test:
	$(MAKE) -C cpp_impl test

test-cpp:
	$(MAKE) -C cpp_impl test-cpp

test-python:
	$(MAKE) -C cpp_impl test-python

sprt:
	$(MAKE) -C cpp_impl sprt

verify:
	$(MAKE) -C cpp_impl verify

cg:
	$(MAKE) -C cpp_impl cg

roundrobin:
	$(MAKE) -C cpp_impl roundrobin

opening-book:
	$(MAKE) -C cpp_impl opening-book

book-inspect:
	$(MAKE) -C cpp_impl book-inspect

clean:
	$(MAKE) -C cpp_impl clean
