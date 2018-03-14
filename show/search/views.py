# -*-coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render_to_response

import random
import model
import codecs

# Create your views here.
def search(request):

	# searching 
	# return HttpResponse(key)
	label = [0,1,2,3,4]
	return render_to_response("search.html", {"random": random.randint(0, 1000)})

def answer(request):
	inquestion = request.POST.get("inputquestion","")
	inarticle = request.POST.get("inputarticle","")
	model.run(inquestion,inarticle)
	ans = []
	f = codecs.open('/Users/Shared/graduation_project/paper/webshow/show/search/data/answers_here.txt','r','utf-8')
	for line in f:
		lin = line.split('\t')
		while '' in lin:
			lin.remove('')
		ans.append({"content":lin[1],"score":lin[0]})

	# if error:
	# 	return HttpResponse("ERROR")

	#ans = [{"content":"123","score":10}, {"content":"456", "score": 43},{"content":inquestion,"score":786},{"content":inarticle,"score":774}]
	return render_to_response("answers.html", {"answer":ans})