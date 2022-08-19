from selenium import webdriver
from time import sleep

driver = webdriver.Firefox()

driver.get("https://bbs.51testing.com/forum.php")
sleep(3)

# 使用ID定位账号输入框 并输入账号
driver.find_element_by_id("ls_username").send_keys("user1")

driver.find_element_by_id("ls_username").send_keys("userpass")

# 定位登录按钮并获取登录按钮的文本
txt = driver.find_element_by_xpath('//*[@id="lsform"]/div/div[1]/table/tbody/tr[2]/td[3]/button').text

# 打印获取到的文本
print(txt)

# 定位“登录”按钮并获取登录按钮的type属性值
type = driver.find_element_by_xpath('//*[@id="lsform"]/div/div[1]/table/tbody/tr[2]/td[3]/button').get_attribute("type")


# 定位“登录”按钮并进行点击操作
driver.find_element_by_xpath('//*[@id="lsform"]/div/div[1]/table/tbody/tr[2]/td[3]/button').click()