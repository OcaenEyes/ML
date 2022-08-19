from selenium import webdriver

driver = webdriver.Firefox()
iframe = driver.find_element_by_xpath()

# 切换到iframe
driver.switch_to.frame(iframe)

# 跳出iframe
driver.switch_to.default_content()
