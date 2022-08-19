from selenium import webdriver
from time import sleep

driver = webdriver.Chrome()
driver.get("https://mail.163.com")
sleep(3)
frame1 = driver.find_element_by_xpath("/html[1]/body[1]/div[2]/div[3]/div[1]/div[1]/div[4]/div[1]/div[1]/iframe[1]")
frame2 = driver.find_element_by_xpath('//*[@id="loginDiv"]/iframe')

driver.switch_to.frame(frame1)
nameplcae=driver.find_element_by_xpath('//*[@id="account-box"]/div[2]/input').get_attribute("data-placeholder")
print(nameplcae)
assert nameplcae == '邮箱帐号或手机号码'

passplace= driver.find_element_by_xpath('//*[@id="login-form"]/div/div[3]/div[2]/input[2]').get_attribute("data-placeholder")
print(passplace)
assert passplace == '输入密码'

driver.find_element_by_xpath('//*[@id="account-box"]/div[2]/input').send_keys("")
driver.find_element_by_xpath('//*[@id="login-form"]/div/div[3]/div[2]/input[2]').send_keys("")
driver.find_element_by_xpath('//*[@id="dologin"]').click()

# 切换出iframe
driver.switch_to.default_content()

sleep(3)
name = driver.find_element_by_xpath("//*[@id='dvContainer']/div/div/div[2]/div/div[2]/span/span").text
print(name)