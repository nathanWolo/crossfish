.PHONY: test test-cpp test-python sprt verify cg cg-input cg-native cg-native-check roundrobin opening-book book-inspect clean

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

cg-input:
	$(MAKE) -C cpp_impl cg-input

cg-native:
	$(MAKE) -C cpp_impl cg-native

cg-native-check:
	$(MAKE) -C cpp_impl cg-native-check

roundrobin:
	$(MAKE) -C cpp_impl roundrobin

opening-book:
	$(MAKE) -C cpp_impl opening-book

book-inspect:
	$(MAKE) -C cpp_impl book-inspect

clean:
	$(MAKE) -C cpp_impl clean
