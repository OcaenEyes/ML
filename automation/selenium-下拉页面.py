from selenium import webdriver
from time import sleep

driver = webdriver.Firefox()

driver.get("http://oceaneyes.top")

sleep(3)

# 页面下拉指定高度
js = 'document.documentElement.scrollTop=800'
driver.execute_script(js)