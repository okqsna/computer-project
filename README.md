# Бібліотека для роботи з графами 
### Комп'ютерний проєкт з дискретної математики 

<i>Над проєктом працювали: Юрій Фітьо, Катерина Сучок, Дарина Ничипорук, Юліан-Володимир Заєць, Оксана Москв'як.</i>

Дана бібліотека опрацьовує отримані на вхід графи з файлів з розширенням .dot,<br>
застосовує до них Гамільтоновий цикл, цикл Ейлера, 
перевіряє на двочастковість, <br> перевіряє на ізоморфність 
та розфарбовує зв'язний граф у 3 кольори.

Детальніше кожну із функцій розглянуто нижче:

### Зчитування файлу, функція ```readfile```
Функція приймає на ввід назву файлу для зчитування. Файл з розширенням .dot, у якому записаний орієнтований чи неорієнтований граф, чиї вершини можуть бути описаними як числами, так і словами. Декілька зв'язків можуть бути записані в один рядок, при чому для орієнтованого вершини зв'язок описується, як "->", а для неорієнтованого "--". Граф описується списком кортежів, якщо граф був неорієнтованим, то цей список перетобляється на список множин. Кожна знайдена вершина записується сет, що перетворюється на список, щоб уникнути повтоорів, а далі списки кортежів чи сетів змінюються так, щоб змінити назву вершини на її порядковий номер у опередньому списку. Таким чином можна називати вершини словами, а працювати з числами, що полегшить роботу наступних функцій. Для того щоб не втрачати інформацію про назви вершин, у функцію передається необов'язковий булівський елемент extra_list, якщо обрати його значення True, то функція повертатиме список, у яком перший елемент - шшуканий список кортежів/масивівб а другий - список вершин, за яким можна відновити назви.

<hr>

### Цикл Ейлера

Спершу йде перевірка на те, чи це орієнтований чи неорієнтований граф. 
Якщо він є неорієнтованим(ребра представлені множинами), то у новий список додається шлях в одну та іншу сторону.
Якщо він орієнтований(ребра представлені кортежами), то іде тоді перевірка, чи цей орієнтований граф є зв'язним.

#### Рекурсивна функція ```calculate_way```.

Вона починає рухатись від початкової вершини, і перебирає варіанти куди можна рухатись далі з точки, в якій ми знаходимось. Так вона генерує всі можливі шляхи. 
Коли ж шлях попадає у вершину, звідки й виходив, та довжина цього шляху є такою, як довжина початкового графу(тобто він містить всі ребра), то він записується до списку, який містить всі можливі цикли ейлера.

Далі ми перебираєм список всіх циклів ейлера і відсіюєм всі зайві цикли, що або повторються, або якщо є їх дзеркальна копія(це той самий цикл, просто в іншому напрямку).

<hr>

### Гамільтоновий цикл
#### Функція ``` check_for_ham ``` 
Ця програма перевіряє, чи має граф Гамільтоновий цикл, і відображає результати у вікні Tkinter, дозволяючи вводити графи у вигляді множин або кортежів.

```permute(nodes)``` Генерує всі перестановки списку вершин. 
```check_for_ham(graph) ``` Перевіряє наявністі гамільтонового циклу в графі.
```parse_input(text)``` Перетворює рядок у список або кортеж. 
```display_results(permutations, correct_path, indx=0)``` Виводить результати перевірки у прокручуваному полі в ткінтері 
```on_enter(event)``` Обробляє натискання клавіші Enter для запуску перевірки циклу
в ткінтерному форматі
```tkinter_window()``` Головну функція Ткінтера, викликаючи її запустить візуалізацію ткінтера

### Алгоритм розв'язування
На початку програма вносить всі вершини графа в словник для збереження з'єднань між ними. Потім я генерую всі можливі перестановки (варіації) множини вершин графа. Для кожної перестановки перевіряю, чи відповідає вона збереженим з'єднанням у словнику. Якщо хоча б одна перестановка проходить перевірку, це означає, що знайдена правильна послідовність вершин для гамільтонового циклу, який проходить через усі вершини графа і повертається до початкової. Якщо жодна перестановка не підходить, повертаю стрінг що про це свідчить.

<hr>

###  Розфарбування зв'язного графу в 3 кольори

#### Функція ```to_matrix``` перетворює відношення на матрицю. 
Відношення може бути заданим як у вигляді списку кортежів, так і у вигляді списку множин, тому для початку функція зводить все до першого варіанту. Якщо ж відношення задане у вигляді списку кортежів, легко створити заповнену нулями матрицю розміром максимального знайденого елементу у кортежіх списку, а потім присвоїти кожному елементу матриці значення 1(True) або 0(False) виразу, що перевіряє, чи існує кортеж з такими координатами в початковому списку. Матриця - список списків.

#### Функція ```to_symetric``` перетворює симетричну матрицю. 
Кожному елементу матриці присвоюється результат побітового або його із симетричним елементом.

#### Функція ```approp``` перевіряє, чи можна зафарбувати вершину конкретним кольором. 
На вхід отримуються номер вершини, матриця, список кольорів усіх вершин (за кольори сприймаються натуральні числа) та пропонований колір вершини (номер). Якщо жодна із сусідніх вершин не має цього кольору, то повертається 1 інакше 0. Сусідніми вершинами вважаються ті, що у матриці мають одинички у одному рядку чи стовпцю із опитуваною вершиною.

#### Функція ```visualizing``` виконує "візуалізацію" алгоритму.
Функція спрацьовуватиме при кожному підборі кольору. Вона перезаписує вихідний файл із частотою в 0.89с, використовуючи бібліотеу time, щоб зміни були помітні. Перезапис полягає в додаванні нового рядка, який міститиме нову інформацію про колір певної вершини. вихідний файл завчасно сформований і містить усю ту інформацію, що була у вхідному файлі.Для цього на вхід передаються ткі параметри: назва вихідного файлуб номер вершини, що записуватиметься, номер її кольору, список кольорів та вершин, щоб можна було відновити інформацію про колір та вершину.

#### Функція ```colouring``` створює числовий список присвоєних кольорів.
На вхід функція отримує матрицю, її розмір, чисельний список присвоєних кольорів вершинам, список назв кольорів, номер вершини, що буде зафарбовуватися, параметр, що відповідає за те, чи потрібна візуалізація та список назв вершин. Ця функція рекурсивна і закінчуватиметься, коли ми дойдемо до останньої вершини, тому на початку відбувається перевірка на те, чи поточна вершина остання, якщо це правда по повертається True. Колір вершини визначаємо підбором, використовуючи для цього цикл. Всередині викликається ```approp```, якщо колір підходить, він записується у чисельний список. Далі ми викликаємо нашу функцію і перевіряємо чи вона повертає значенння True. По суті, в цьому моменті реалізовується рекурсія, яка закінчиться, коли ми дійдемо до останнього елементу. Коли це станеться, функція поверне заповнений чисельний список. Якщо ж у якісь вешині цикл дійде до кінця і не зможе підібрати жодного кольору, то функція поверне False. Також ппри кожній ітерації, якщо візуалізація потрібна, викликається функція ```visualizing```.

#### Функція ```get_colour_seq``` об'єднує усі роботу усіх попередніх функцій та повертає стрічку з набором кольорів, або стрічку з повідомленням про неможливість розмальовування.
На вхід функція отримує назву файлу з графом із розширенням .dot, список кольорів, що ми можемо використовувати для розмальовування, параметр, що відповідає за необхідніть візуалізації та назву вихідного файлу. Всередині функція створює симетричну матрицю, знаходить її розмір, формує заповнений нулями чисельний список для розмальовування розміром в кількість вершин, викликає функцію ```colouring``` та перевіряє чи вона не повертає нуль, тобто чи можна розмалювати граф таким чином. Якщо не можна, то у результат записується повідомлення: "The colouring is imposible.", інакше результат виклику функції зберігається та формується стрічка із списком кольорів через пробіл, яку функція повертатиме.

#### Функція ```write_colour``` створює новий файл, куди записує розмальований граф.
На вхід отримуються назви вхідного та вихідного файлу, список назв кольорів та параметр, що передається за замовчуванням, що відповідає за візуалізацію. Якщо останній параметр передається, як True, то завчасно створюється вихідний файл, як копія вхідного. Спочатку зберігається результат виклику функції ```get_color_seq```. Якщо ми отримали повідомлення про неможливість розмалювання, воно виводиться на екран, інакше відкриваються файли для читання та запису. Із вхідного у вихідний переписується все окрім останнього рядку, в якому записаний знак "}". Далі знаходиться їхня кількість і прописується опис кожної вершини, де вказується колір, який береться по порядку із отриманої стрічки.

<hr>

### Двочастковість графу

#### Функція ``` bipartite_graph_check ``` перевіряє чи граф є двочастковим.

Основною ідеєю реалізації даної функції є використання <b>алгоритму пошуку в ширину (BFS)</b> та <b>розфарбовування вершин графа у два кольори</b> (тобто реалізація розділення вершин графа на 2 множини, у яких не повторюються вершини - основна умова двочастковості графа). 

Допоміжна функція ``` convert_to_directed ``` перетворює граф з неорієнтованого в орієнтований для подальшої роботи.

Допоміжна функція ``` get_neighbouring_values ``` знаходить для кожної вершини суміжні та повертає словник, де ключ це <b>вершина</b>, а значення - <b>множина суміжних вершин</b>.

На початку функція створює <b>“чергу”</b> ``` not_visited_vertices ``` для зафарбовування вершин із цієї черги та словник, який зберігає колір вершин. Імплементовано цикл ``` while ```, який працює поки черга не буде пустою. Для подальшої перевірки дістаємо першу вершину з черги, для якої відбувається перевірка чи не є ця вершина в відвіданих, далі відбувається перевірка суміжних до початкової вершин. Якщо поточна вершина та суміжна їй вершини однакового кольору - то цей граф <b>не двочастковий</b>, функція повертає значення - ``` False ```

Якщо всі вершини було зафарбовано без збігів, то граф <b>двочастковий</b>. Функція повертає значення - ``` True ```

<hr>

### Ізоморфність графу

#### Функція ```if_graphs_are_isomorphic``` перевіряє чи два графи є ізоморфними.

Допоміжні функції ```if_graph_is_directed``` і ```if_graph_is_undirected``` перевіряють чи графи <b>є орієнтованими, чи є неорієнтованими.</b>

Допоміжна функція ```directed_isomorphism``` перевіряє <b>ізоморфність</b> двох орієнтованих графів.

Алгоритм базується <b> на порівнянні структур графів шляхом перебору всіх можливих перестановок вершин.</b><br>
Спершу визначають списки вершин обох графів. Якщо кількість вершин у графах <b>відрізняється</b>, вони <b>не можуть</b> бути ізоморфними. Далі для кожної перестановки вершин перевіряється, чи відповідає структура одного графа структурі іншого, шляхом порівняння матриць суміжності або списків суміжності, враховуючи напрямки ребер. <br>
Алгоритм завершується та повертає ```True```, якщо знайдена відповідність, або повертає ```False``` після перевірки всіх можливих перестановок.

Допоміжна функція ```undirected_isomorphism``` перевіряє ізоморфність двох неорієнтованих графів.<br>

Алгоритм схожий до попередньої функції, але додатково <b>враховує симетричність</b> зв’язків між вершинами. Під час порівняння списків суміжності перевіряється, чи присутнє кожне ребро між вершинами в обох напрямках, оскільки у неорієнтованих графах порядок вершин у ребрі не має значення. Якщо <b>знаходиться перестановка</b>, яка забезпечує однаковість структур обох графів, вони вважаються <b>ізоморфними</b>; інакше — ні.

<hr>



#### Розподіл обов'язків
Під час роботи над цим комп'ютерним проєктом ми розприділили функції по одній для кожного, проте ми допомогали один одному, підказували, як можна покращити нашу бібліотеку та постійно вели комунікацію про це в команді.
#### Враження від проєкту
Під час роботи над цим комп'ютерним проєктом ми отримали нові знання з розділу дискретної математики - графи, познайомились з різноманітними алгоритмами для безпосередньої роботи з графами та покращили навики роботи з різноманітними бібліотеками мови програмування Python. <br>
Даний проєкт став для нас відкриттям, оскільки ми працювали над алгоритмами, які можна застосувати не лише для теоретичної роботи з графами, але й для вирішення завдань у реальному житті. 
 <br>
Він також подарував нам цікавий досвід роботи в команді та показав, що працюючи спільно можна знайти ефективні та неочевидні вирішення поставлених завдань, і найважливіше - створити ефективну та корисну програму.
