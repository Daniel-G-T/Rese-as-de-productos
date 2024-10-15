import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from wordcloud import WordCloud
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression


###################
#Funciones
en_stop = {'a','al','algo','algunas','algunos','ante','antes','como','con','contra','cual','cuando','de','del','desde','donde','durante','e','el','ella','ellas','ellos','en','entre','era','erais','eran','eras','eres','es','esa','esas','ese','eso','esos','esta','estaba','estabais','estaban','estabas','estad','estada','estadas','estado','estados','estamos','estando','estar','estaremos','estará','estarán','estarás','estaré','estaréis','estaría','estaríais','estaríamos','estarían','estarías','estas','este','estemos','esto','estos','estoy','estuve','estuviera','estuvierais','estuvieran','estuvieras','estuvieron','estuviese','estuvieseis','estuviesen','estuvieses','estuvimos','estuviste','estuvisteis','estuviéramos','estuviésemos','estuvo','está','estábamos','estáis','están','estás','esté','estéis','estén','estés','fue','fuera','fuerais','fueran','fueras','fueron','fuese','fueseis','fuesen','fueses','fui','fuimos','fuiste','fuisteis','fuéramos','fuésemos','ha','habida','habidas','habido','habidos','habiendo','habremos','habrá','habrán','habrás','habré','habréis','habría','habríais','habríamos','habrían','habrías','habéis','había','habíais','habíamos','habían','habías','han','has','hasta','hay','haya','hayamos','hayan','hayas','hayáis','he','hemos','hube','hubiera','hubierais','hubieran','hubieras','hubieron','hubiese','hubieseis','hubiesen','hubieses','hubimos','hubiste','hubisteis','hubiéramos','hubiésemos','hubo','la','las','le','les','lo','los','me','mi','mis','mucho','muchos','muy','más','mí','mía','mías','mío','míos','nada','ni','no','nos','nosotras','nosotros','nuestra','nuestras','nuestro','nuestros','o','os','otra','otras','otro','otros','para','pero','poco','por','porque','que','quien','quienes','qué','se','sea','seamos','sean','seas','sentid','sentida','sentidas','sentido','sentidos','seremos','será','serán','serás','seré','seréis','sería','seríais','seríamos','serían','serías','seáis','siente','sin','sintiendo','sobre','sois','somos','son','soy','su','sus','suya','suyas','suyo','suyos','sí','también','tanto','te','tendremos','tendrá','tendrán','tendrás','tendré','tendréis','tendría','tendríais','tendríamos','tendrían','tendrías','tened','tenemos','tenga','tengamos','tengan','tengas','tengo','tengáis','tenida','tenidas','tenido','tenidos','teniendo','tenéis','tenía','teníais','teníamos','tenían','tenías','ti','tiene','tienen','tienes','todo','todos','tu','tus','tuve','tuviera','tuvierais','tuvieran','tuvieras','tuvieron','tuviese','tuvieseis','tuviesen','tuvieses','tuvimos','tuviste','tuvisteis','tuviéramos','tuviésemos','tuvo','tuya','tuyas','tuyo','tuyos','tú','un','una','uno','unos','vosotras','vosotros','vuestra','vuestras','vuestro','vuestros','y','ya','yo','él','éramos'}

#######################
# Page configuration
st.set_page_config(
    page_title="Opiniones textuales de productos",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)

#######################
#Exportación de datos
datos = pd.read_csv('Review.csv')
productos = {0:'coches', 1:'hoteles', 2:'lavadoras', 3:'libros', 4:'moviles', 5:'musica', 6:'ordenadores',
           7:'peliculas'}


token_length = [len(t.split()) for t in datos.Tokens]
datos['token_length'] = token_length


##################################################
# Dashboard Main Panel

st.title("Machine learning en textos de reseñas")
tab1, tab2 , tab3= st.tabs(["Descripción", "Visualización","Clasificación"])

with tab1:
	st.header("Datos de la base de datos", divider="gray")
	st.markdown("El conjunto de datos empleado en este ejercicio corresponde a diversas opiniones de usuarios en los siguientes productos: automóviles, hoteles, lavadoras, libros, teléfonos celulares, música, computadoras y películas. Este conjunto se basa en 400 opiniones junto con su categoría y sentimiento. Para el sentimiento tenemos dos categorias: yes y no, que indican las opiniones positivas y negativas, respectívamente. En esta base de datos las opiniones, 200 opiniones positivas y las otras son negativas mientras que cada articulo tiene un total de 50 registros.")

	text_length = [len(t) for t in datos.Texto]
	st.write("Promedio de tokens por opinión: "+ str(round(np.mean(text_length),4)))
	fig = go.Figure(data=[go.Histogram(x=text_length)])
	fig.update_layout(title='Longitud de opiniones',
    		xaxis=dict(title='Longitud de opinión',titlefont_size=16,tickfont_size=14),
    		yaxis=dict(title='Frecuencia',titlefont_size=16,tickfont_size=14))
	st.plotly_chart(fig, use_container_width=True) 

	fig2 = px.bar(datos, x='categoria', y='token_length', color='sentimiento')
	fig2.update_layout(title='Tokens por producto',
    		xaxis=dict(title='Producto',titlefont_size=16,tickfont_size=14),
    		yaxis=dict(title='Cantidad de tokens',titlefont_size=16,tickfont_size=14))
	st.plotly_chart(fig2, use_container_width=True) 
	
	st.subheader("Nube de palabras de tokens")
	st.markdown("En esta nube de palabras, se destacan las principales percepciones de nuestros clientes sobre el producto. Las palabras más grandes representan los términos más repetidos y reflejan las características o experiencias que más resonaron con nuestros usuarios. Esto nos ofrece una visión rápida de las opiniones generales, desde los aspectos más valorados hasta los puntos de mejora mencionados con frecuencia.")
	nube_image = WordCloud(collocations =False,max_words=20,background_color="white",min_word_length=3,colormap='OrRd',
                 stopwords=['parecer','mismo','decir','solo','primero','aunque','hacer','poder']+['hotel','coche','lavadora','libro',
		'película']).generate(" ".join(datos.Tokens))

	fig3 = go.Figure()
	fig3.add_trace(go.Image(z=nube_image))
	fig3.update_xaxes(visible=False, fixedrange=False)
	fig3.update_yaxes(visible=False, fixedrange=False)
	st.plotly_chart(fig3, use_container_width=True) 

with tab2:
	st.header("Representación de textos con TSNE", divider="gray")
	vectorizer = TfidfVectorizer(lowercase=False, ngram_range= (1,1), max_features=750, binary=False)
	X = vectorizer.fit_transform(datos.Tokens)	
	tf_idf = X.toarray()
	tf_idf_df = pd.DataFrame(tf_idf,columns=vectorizer.get_feature_names_out())
	
	option = st.selectbox("Selecciona un periodo de conteo:",("TSNE", "LLE", "PCA"),)
	if option == "TSNE":
		tsne = TSNE(init = 'random', perplexity=30, n_iter_without_progress=150, n_jobs=2, random_state=0)
		emb = tsne.fit_transform(tf_idf_df)
	else:
		embedding = LocallyLinearEmbedding(n_components=2)
		emb = embedding.fit_transform(tf_idf_df)
	
	proj = pd.DataFrame(emb,columns = ['pc1','pc2'])
	fig4 = px.scatter(proj, x="pc1", y="pc2", color=datos['categoria'])
	fig4.update_traces(marker_size=8)
	st.plotly_chart(fig4, use_container_width=True) 

with tab3:
	st.header("Clasificación de texto", divider="gray")
	st.markdown("Resultados al clasificar 80 opiniones (10 de cada producto) con un modelo de regresión logística entrenado en el resto de opiniones.")
	X_train, X_test, y_train, y_test = train_test_split(tf_idf_df, datos['categoria'], test_size=0.20, random_state=20)
	reg = LogisticRegression(fit_intercept=True,random_state=0, multi_class='multinomial').fit(X_train, y_train)
	y_pred=reg.predict(X_test)
	correct = (y_pred == y_test)

	fig5 = px.scatter(proj.iloc[X_test.index,:][correct], x="pc1", y="pc2", 
                 color=datos['categoria'][X_test.index][correct])
	fig5.add_trace(go.Scatter(mode='markers',x=proj.iloc[X_test.index,0][~correct],y=proj.iloc[X_test.index,1][~correct],
		marker=dict(color='darkred',size = 20, symbol='cross'),name = "Errores",showlegend=True,))
	st.plotly_chart(fig5, use_container_width=True) 
	st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())
	
