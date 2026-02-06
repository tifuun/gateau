#!/bin/sh

sleep 3
xclip -o -selection CLIPBOARD | xdotool type --file -

