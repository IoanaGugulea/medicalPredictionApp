import time
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Inițializează WebDriver
driver = webdriver.Chrome()

try:
    # Deschide site-ul IQAir
    url = "https://www.iqair.com/romania/ilfov/mogosoaia"
    driver.get(url)
    wait = WebDriverWait(driver, 15)
    
    # Scroll în jos pentru a încărca datele complet
    for _ in range(5):
        driver.execute_script("window.scrollBy(0, 500);")
        time.sleep(2)  # Pauză pentru încărcarea conținutului

    # Salvează screenshot pentru debugging
    driver.save_screenshot("debug_screenshot.png")
    print("Screenshot salvat pentru debugging: debug_screenshot.png")
    
    # Găsește tabelul principal cu date
    forecast_table = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.table-wrapper table")))
    rows = forecast_table.find_elements(By.TAG_NAME, "tr")

    # Creează fișier CSV pentru date
    csv_filename = "pm25_data.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "PM2.5 (µg/m³)", "Precipitation (%)", "Temperature (°C)", 
                         "Temperature (Wind)", "Wind Speed (km/h)", "Humidity (%)"])

        # Iterează prin rânduri și extrage datele
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) >= 6:
                date = cells[0].text.strip().replace("\n", " ")
                pm25 = cells[1].text.strip().replace(" µg/m³", "").replace("\n", " ")
                
                # Precipitation - verificăm dacă există valoare pentru precipitații
                precipitation = cells[2].text.strip().replace("\n", " ")
                if not precipitation or precipitation == "-":
                    precipitation = "0%"  # Dacă lipsesc datele de precipitații, setăm la 0%
                else:
                    if "%" not in precipitation:
                        precipitation = precipitation + "%"  # Dacă nu există simbolul %, îl adăugăm
                
                # Separarea valorilor de Temperatură și Viteza Vântului
                temp_wind = cells[3].text.strip().replace("\n", " ")
                if " " in temp_wind:
                    temp, wind_speed = temp_wind.split(" ", 1)
                    wind_speed = wind_speed.replace(" km/h", "").strip()
                else:
                    temp = temp_wind
                    wind_speed = "0"  # Dacă lipsește viteza vântului, o setăm la 0
                
                # Dacă temperatura este goală, o setăm la 0
                temp = temp if temp else "0"

                # Scrie datele în fișier în ordinea dorită
                writer.writerow([date, pm25, precipitation, temp, temp_wind, wind_speed, cells[5].text.strip().replace("%", "").replace("\n", " ")])
    
    print(f"Datele au fost salvate în fișierul {csv_filename}")

finally:
    driver.quit()
