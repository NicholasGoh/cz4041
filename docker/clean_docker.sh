#!/bin/bash

echo "Deleting stopped containers ..."
sudo docker rm $(sudo docker ps -qa --no-trunc --filter "status=exited")

echo "Deleting stale images ..."
sudo docker rmi $(sudo docker images --filter "dangling=true" -q --no-trunc)
