#!/bin/bash

#THreshold values
CPU_THRESHOLD=80
MEMORY_THRESHOLD=80
DISK_THRESHOLD=80

#Function to send alert
send_alert() {
    echo "$(tput setaf 1)ALERT: $1 usage exceeded threshold! Current value: $2%$(tput sgr0)"
}
# send_alert "CPU" 85


while true; do 
    #CPU usage 
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
    cpu_usage=${cpu_usage%.*} #Integer convertion
    echo "Current CPU usage: $cpu_usage%"

    if ((cpu_usage >= CPU_THRESHOLD)); then
        send_alert "CPU" "$cpu_usage"
    fi 

    #Memory usage
    memory_usage=$(free | awk '/Mem/ {printf("%3.1f", ($3/$2) * 100)}')
    echo "Current memory usage: $memory_usage%"
    memory_usage=${memory_usage%.*}
    if ((memory_usage >= MEMORY_THRESHOLD)); then
        send_alert "Memory" "$memory_usage"
    fi 

    #Disk usage
    disk_usage=$(df -h / | awk '/\// {print $(NF-1)}')
    disk_usage=${disk_usage%?}
    echo "Current disk usage: $disk_usage%"

    if ((disk_usage >= DISK_THRESHOLD)); then
        send_alert "Disk" "$disk_usage"
    fi 

    clear 
    echo "Resource Usage: "
    echo "CPU: $cpu_usage%"
    echo "Memory: $memory_usage%"
    echo "Disk: $disk_usage%"
    sleep 2
done