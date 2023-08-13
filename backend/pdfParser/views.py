# from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from AIModel import main

@api_view(['POST'])
def askQuery(request):
    # main.downloadFile()
    result = main.loadModel(request.data['data'])
    return Response({'result':result}, status=status.HTTP_200_OK)